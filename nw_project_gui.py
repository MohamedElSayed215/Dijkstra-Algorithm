import networkx as nx
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np # For calculating intermediate points for animation

# --- Network Logic Functions ---

def create_network_graph(file_path):
    """
    Reads network topology from a file and creates a networkx graph.
    Reads the number of nodes and edges from the first line.
    Reads edges (src, dest, weight) from subsequent lines.

    Args:
        file_path (str): The path to the input file.

    Returns:
        tuple: A tuple containing:
               - nx.Graph: The created graph object or None if file not found or parsing error occurs.
               - str: An error message string, or None on success.
    """
    G = nx.Graph() # Use nx.DiGraph() for directed edges if needed
    try:
        with open(file_path, 'r') as f:
            # Read number of nodes and edges from the first line
            header = f.readline().split()
            if len(header) != 2:
                 return None, f"Error reading header in {file_path}. Expected 'num_nodes num_edges'. Got: {' '.join(header)}"
            try:
                num_nodes = int(header[0])
                num_edges = int(header[1])
            except ValueError:
                 return None, f"Error: Could not parse number of nodes/edges from header: {' '.join(header)}"


            # Add edges with weights
            added_edges = 0
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    src, dest, weight_str = parts
                    try:
                        weight = int(weight_str.strip())
                        G.add_edge(src.strip(), dest.strip(), weight=weight)
                        added_edges += 1
                    except ValueError:
                         print(f"Warning: Skipping line with invalid weight format: {line.strip()}", file=sys.stderr) # Keep warnings in console
                    except Exception as e:
                         print(f"Warning: Error adding edge from line {line.strip()}: {e}", file=sys.stderr) # Keep warnings in console
                else:
                     print(f"Warning: Skipping malformed line: {line.strip()}. Expected 'src,dest,weight'", file=sys.stderr) # Keep warnings in console

            if added_edges != num_edges:
                 print(f"Warning: Expected {num_edges} edges but added {added_edges}.", file=sys.stderr) # Keep warnings in console

        return G, None # Return graph and no error message on success
    except FileNotFoundError:
        return None, f"Error: Input file '{file_path}' not found."
    except Exception as e:
        return None, f"An unexpected error occurred while reading file {file_path}: {e}"


def generate_forwarding_tables(graph):
    """
    Generates forwarding tables for all nodes in the graph using shortest paths
    calculated by Dijkstra's algorithm (implemented in networkx).

    Args:
        graph (nx.Graph): The network graph.

    Returns:
        tuple: A tuple containing:
               - dict: A dictionary where keys are source nodes and values are their forwarding tables (dict).
               - str: A string containing formatted forwarding tables, or an error message.
    """
    all_forwarding_tables = {}
    output_text = ""

    if graph is None:
        return all_forwarding_tables, "Error: Graph is not created."

    output_text += "--- Generating Forwarding Tables ---\n"
    for source_node in graph.nodes():
        try:
            # Calculate shortest paths from the current source node to all others
            shortest_paths_from_node = nx.shortest_path(graph, source=source_node, weight='weight')

            forwarding_table = {}
            # Iterate through all possible destinations in the graph
            for destination in graph.nodes():
                 if destination in shortest_paths_from_node:
                    path = shortest_paths_from_node[destination]
                    if destination == source_node:
                        forwarding_table[destination] = "-" # Or indication of direct connection/self
                    elif len(path) > 1:
                        forwarding_table[destination] = path[1] # The next hop is the second node in the shortest path
                    else:
                        forwarding_table[destination] = destination # Direct neighbor
                 else:
                     forwarding_table[destination] = "Unreachable" # Indicate unreachable

            all_forwarding_tables[source_node] = forwarding_table

        except nx.NetworkXNoPathError:
            output_text += f"Warning: Node {source_node} cannot reach all other nodes in the graph.\n"
            forwarding_table = {}
            for dest_node in graph.nodes():
                 if dest_node == source_node:
                     forwarding_table[dest_node] = "-"
                 else:
                     forwarding_table[dest_node] = "Unreachable"
            all_forwarding_tables[source_node] = forwarding_table

        except Exception as e:
            output_text += f"An unexpected error occurred while generating table for node {source_node}: {e}\n"


    # Format the tables for display in the GUI
    for node, table in all_forwarding_tables.items():
        output_text += f"\nForwarding Table for node {node}:\n"
        output_text += "Destination | Next Hop\n"
        output_text += "----------|---------\n"
        # Sort table by destination for cleaner output
        for dest in sorted(table.keys()):
             output_text += f"{dest:<11}| {table[dest]}\n"

    return all_forwarding_tables, output_text


def create_network_visualization(graph, pos=None):
    """
    Creates a matplotlib figure and axes for the network graph visualization.
    Optionally takes a predefined layout position.
    Does NOT call plt.show().

    Args:
        graph (nx.Graph): The network graph.
        pos (dict, optional): A dictionary of node positions. If None, a layout is generated.

    Returns:
        tuple: A tuple containing:
               - Figure: The matplotlib figure object.
               - Axes: The matplotlib axes object.
               - dict: The node positions used for the layout.
               - str: An error message, or None on success.
    """
    if graph is None:
        return None, None, None, "Cannot visualize: Graph is None."

    try:
        # Create a figure and axes
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)

        # Generate layout if not provided
        if pos is None:
            pos = nx.spring_layout(graph) # Try different layouts

        # Draw the nodes and edges on the axes
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=800, edge_color='k', linewidths=1, font_size=10, ax=ax)

        # Draw edge weights on the axes
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', ax=ax)

        # Set title
        ax.set_title("Network Topology")

        return fig, ax, pos, None # Return figure, axes, positions, and no error message
    except Exception as e:
        return None, None, None, f"Error during visualization creation: {e}"


# --- GUI Implementation ---

class NetworkApp:
    def __init__(self, root):
        self.root = root
        root.title("Network Link-State Simulator")

        self.graph = None # To store the network graph
        self.pos = None # To store node positions for consistent layout
        self.canvas = None # To store the matplotlib canvas
        self.toolbar = None # To store the matplotlib toolbar
        self.animation_marker = None # To store the animation marker artist

        # Animation state variables for smoother movement
        self.animation_path = [] # List of nodes in the shortest path
        self.current_path_segment_index = 0 # Index of the current edge being animated (0 to len(path)-2)
        self.current_segment_progress = 0.0 # Progress along the current edge (0.0 to 1.0)
        self.segment_steps = 20 # Number of small steps to take along each edge for smoother animation
        self.animation_speed = 50 # Milliseconds delay between each small animation step (adjust for overall speed)
        self._animation_job_id = None # To store the ID returned by root.after()


        # --- GUI Elements ---

        # Main Panes (using PanedWindow for resizable sections)
        self.paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Pane: Controls and Output
        left_pane = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(left_pane, sticky="nsew")

        # File Selection Frame
        file_frame = ttk.LabelFrame(left_pane, text="Input File")
        file_frame.pack(pady=5, fill="x")

        self.file_path_entry = ttk.Entry(file_frame, width=40)
        self.file_path_entry.pack(side="left", padx=5, pady=5, expand=True, fill="x")

        browse_button = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        browse_button.pack(side="left", padx=5, pady=5)

        # Process Button
        process_button = ttk.Button(left_pane, text="Process Network", command=self.process_network)
        process_button.pack(pady=5)

        # Output Area
        output_frame = ttk.LabelFrame(left_pane, text="Output (Graph Info & Forwarding Tables)")
        output_frame.pack(pady=5, fill="both", expand=True)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=60, height=15, font=('Courier New', 10)) # Use a monospace font for tables
        self.output_text.pack(padx=5, pady=5, fill="both", expand=True)

        # Animation Controls Frame
        animation_frame = ttk.LabelFrame(left_pane, text="Animate Shortest Path")
        animation_frame.pack(pady=10, fill="x")

        # Source Node Selection
        source_frame = ttk.Frame(animation_frame)
        source_frame.pack(pady=2, fill="x")
        ttk.Label(source_frame, text="Source Node:").pack(side="left", padx=5)
        self.source_node_combobox = ttk.Combobox(source_frame, state="disabled", width=10)
        self.source_node_combobox.pack(side="left", padx=5, expand=True, fill="x")

        # Destination Node Selection
        dest_frame = ttk.Frame(animation_frame)
        dest_frame.pack(pady=2, fill="x")
        ttk.Label(dest_frame, text="Destination Node:").pack(side="left", padx=5)
        self.dest_node_combobox = ttk.Combobox(dest_frame, state="disabled", width=10)
        self.dest_node_combobox.pack(side="left", padx=5, expand=True, fill="x")

        # Animate Button
        self.animate_button = ttk.Button(animation_frame, text="Animate", command=self.start_animation, state="disabled")
        self.animate_button.pack(pady=5)


        # Right Pane: Visualization
        right_pane = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(right_pane, sticky="nsew")

        self.visualization_frame = ttk.LabelFrame(right_pane, text="Network Visualization")
        self.visualization_frame.pack(pady=5, fill="both", expand=True)

        # Placeholder for the matplotlib canvas
        self.plot_container = ttk.Frame(self.visualization_frame)
        self.plot_container.pack(fill="both", expand=True)


    def browse_file(self):
        """Opens a file dialog to select the input file."""
        file_path = filedialog.askopenfilename(
            initialdir=".", # Start browsing from the current directory
            title="Select Network Input File",
            filetypes=(("Text Files", "*.txt"), ("All files", "*.*"))
        )
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)
            # Disable animation controls until a graph is processed
            self.source_node_combobox['state'] = 'disabled'
            self.dest_node_combobox['state'] = 'disabled'
            self.animate_button['state'] = 'disabled'
            self.source_node_combobox.set('')
            self.dest_node_combobox.set('')


    def process_network(self):
        """Reads the file, creates graph, calculates tables, and visualizes."""
        file_path = self.file_path_entry.get()
        if not file_path:
            messagebox.showwarning("Input Error", "Please select an input file.")
            return

        self.output_text.delete(1.0, tk.END) # Clear previous output
        self.output_text.insert(tk.END, f"Processing network from: {file_path}\n")
        self.output_text.update() # Update GUI to show message

        # --- Step 1: Create the network graph ---
        self.graph, error_message = create_network_graph(file_path)

        if error_message:
            messagebox.showerror("File Reading Error", error_message)
            self.output_text.insert(tk.END, f"Error: {error_message}\n")
            # Disable animation controls on error
            self.source_node_combobox['state'] = 'disabled'
            self.dest_node_combobox['state'] = 'disabled'
            self.animate_button['state'] = 'disabled'
            self.source_node_combobox.set('')
            self.dest_node_combobox.set('')
            self.clear_plot() # Clear any previous plot
            return

        if self.graph:
            self.output_text.insert(tk.END, "\nGraph created successfully!\n")
            self.output_text.insert(tk.END, f"Number of nodes: {self.graph.number_of_nodes()}\n")
            self.output_text.insert(tk.END, f"Number of edges: {self.graph.number_of_edges()}\n")
            nodes = sorted(list(self.graph.nodes())) # Get sorted nodes for comboboxes
            self.output_text.insert(tk.END, f"Nodes: {nodes}\n")
            # Edges output can be verbose, maybe print only if graph is small or add an option
            # self.output_text.insert(tk.END, f"Edges (with weights): {list(self.graph.edges(data=True))}\n")
            self.output_text.update()

            # --- Step 2: Generate and display forwarding tables ---
            all_tables, tables_output_text = generate_forwarding_tables(self.graph)
            self.output_text.insert(tk.END, tables_output_text)
            self.output_text.update()

            # --- Step 3: Create and embed the network visualization ---
            # Clear previous plot if exists
            self.clear_plot()

            fig, ax, self.pos, viz_error = create_network_visualization(self.graph)

            if viz_error:
                 messagebox.showerror("Visualization Error", viz_error)
                 self.output_text.insert(tk.END, f"Visualization Error: {viz_error}\n")
                 # Disable animation controls on visualization error
                 self.source_node_combobox['state'] = 'disabled'
                 self.dest_node_combobox['state'] = 'disabled'
                 self.animate_button['state'] = 'disabled'
                 self.source_node_combobox.set('')
                 self.dest_node_combobox.set('')
            elif fig and ax and self.pos:
                self.embed_plot(fig, ax)
                self.output_text.insert(tk.END, "\nNetwork visualization created.\n")
                self.output_text.update()

                # Enable and populate animation controls
                self.source_node_combobox['values'] = nodes
                self.dest_node_combobox['values'] = nodes
                self.source_node_combobox['state'] = 'readonly' # Prevent typing
                self.dest_node_combobox['state'] = 'readonly' # Prevent typing
                self.animate_button['state'] = 'normal'

        else:
            self.output_text.insert(tk.END, "\nFailed to create network graph.\n")
            # Disable animation controls if graph creation failed
            self.source_node_combobox['state'] = 'disabled'
            self.dest_node_combobox['state'] = 'disabled'
            self.animate_button['state'] = 'disabled'
            self.source_node_combobox.set('')
            self.dest_node_combobox.set('')
            self.clear_plot() # Clear any previous plot


    def embed_plot(self, figure, axes):
        """Embeds a matplotlib figure into the plot_container frame."""
        # Create a canvas widget for the figure
        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_container)
        self.canvas.draw()

        # Create a toolbar for the canvas (optional, but adds interactivity)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_container)
        self.toolbar.update()

        # Pack the canvas and toolbar
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Store axes for animation
        self.ax = axes


    def clear_plot(self):
        """Clears the embedded matplotlib plot and toolbar."""
        # Stop any ongoing animation before clearing
        if self._animation_job_id is not None:
             self.root.after_cancel(self._animation_job_id)
             self._animation_job_id = None
        self.animation_marker = None # Clear the marker reference

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
        # Close any lingering matplotlib figures to free memory
        plt.close('all')


    def start_animation(self):
        """Starts the animation of data moving along the shortest path."""
        if self.graph is None or self.pos is None or self.ax is None:
            messagebox.showwarning("Animation Error", "Network graph not processed yet.")
            return

        source_node = self.source_node_combobox.get()
        dest_node = self.dest_node_combobox.get()

        if not source_node or not dest_node:
            messagebox.showwarning("Animation Error", "Please select both source and destination nodes.")
            return

        if source_node not in self.graph.nodes() or dest_node not in self.graph.nodes():
             messagebox.showwarning("Animation Error", "Selected nodes are not in the graph.")
             return

        try:
            # Calculate the shortest path
            self.animation_path = nx.shortest_path(self.graph, source=source_node, target=dest_node, weight='weight')
            print(f"Animating path: {self.animation_path}") # Debug print

            if len(self.animation_path) < 2:
                 messagebox.showinfo("Animation", f"Source and destination are the same or directly connected with no intermediate steps.")
                 return

            # Initialize animation state variables
            self.current_path_segment_index = 0
            self.current_segment_progress = 0.0

            # Clear previous animation marker if exists
            if self.animation_marker:
                self.animation_marker.remove()
                self.animation_marker = None

            # Start the animation loop
            self.animate_step()

        except nx.NetworkXNoPathError:
            messagebox.showinfo("Animation", f"No path found between {source_node} and {dest_node}.")
        except Exception as e:
            messagebox.showerror("Animation Error", f"An error occurred during animation setup: {e}")


    def animate_step(self):
        """Performs one step of the animation."""
        # Check if we have processed all segments in the path
        if self.current_path_segment_index >= len(self.animation_path) - 1:
            # Animation finished
            if self.animation_marker:
                 self.animation_marker.remove() # Remove the marker at the end
                 self.animation_marker = None
            self.canvas.draw_idle() # Update canvas to remove the marker
            print("Animation finished.") # Debug print
            self._animation_job_id = None # Clear the job ID
            return

        # Get current and next node in the current segment
        node1 = self.animation_path[self.current_path_segment_index]
        node2 = self.animation_path[self.current_path_segment_index + 1]

        # Get positions of current and next node
        p1 = self.pos[node1]
        p2 = self.pos[node2]

        # Calculate interpolation factor based on progress along the segment
        interp_factor = self.current_segment_progress

        # Calculate marker position using linear interpolation
        marker_x = p1[0] + (p2[0] - p1[0]) * interp_factor
        marker_y = p1[1] + (p2[1] - p1[1]) * interp_factor

        # Clear previous marker if exists
        if self.animation_marker:
            self.animation_marker.remove()

        # Draw the new marker
        # 'o' is a circle marker, 'r' is red color, markersize controls size
        # Use the ax object stored during embed_plot
        self.animation_marker, = self.ax.plot(marker_x, marker_y, 'o', color='red', markersize=10, zorder=5) # zorder ensures marker is on top

        # Redraw the canvas
        self.canvas.draw_idle() # Use draw_idle for efficiency

        # Increment progress along the current segment
        self.current_segment_progress += 1.0 / self.segment_steps

        # Check if we finished the current segment
        if self.current_segment_progress >= 1.0:
            # Move to the next segment
            self.current_path_segment_index += 1
            self.current_segment_progress = 0.0 # Reset progress for the new segment

        # Schedule the next animation step
        # Store the returned ID so we can cancel it if needed
        self._animation_job_id = self.root.after(self.animation_speed, self.animate_step)


# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    # Maximize the window
    root.state('zoomed') # Use 'zoomed' for Windows, 'normal' then root.attributes('-zoomed', True) for Linux/X11

    app = NetworkApp(root)
    root.mainloop()
