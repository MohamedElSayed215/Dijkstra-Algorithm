import networkx as nx
import matplotlib.pyplot as plt
import heapq
import sys
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np # For calculating intermediate points for animation

# --- Network Logic Functions ---

# Modified dijkstra function to yield steps
def dijkstra_step_by_step(graph, start):
    """
    Runs Dijkstra's algorithm step-by-step and yields the state at each step.

    Args:
        graph (nx.Graph): The network graph.
        start (str): The starting node.

    Yields:
        tuple: (step_count, N_prime, current_distances, current_previous)
               - step_count (int): The current step number.
               - N_prime (set): The set of nodes whose shortest path is finalized.
               - current_distances (dict): Current shortest distance estimates for all nodes.
               - current_previous (dict): Current predecessors for path reconstruction.
    """
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start] = 0
    previous = {}
    queue = [(0, start)]
    n_prime = set() # Set of nodes whose shortest path is finalized
    step_count = 0

    # Initial state (Step 0)
    yield step_count, set(n_prime), dict(distances), dict(previous) # Yield copies

    while queue:
        # Find the node with the minimum distance estimate among nodes not in N'
        min_dist = float('inf')
        next_node_to_finalize = None
        # Iterate through all nodes to find the one with min distance not in N'
        # (This is a simplified approach, a priority queue handles this efficiently in the main dijkstra)
        # However, for step-by-step visualization, iterating through all nodes to show D values is needed.
        # Let's stick to the heapq approach for correct algorithm logic, but ensure we only process
        # the node popped from the queue if its distance hasn't been improved.

        dist, node = heapq.heappop(queue)

        # If already found a shorter path or node is already finalized, skip
        if dist > distances[node] or node in n_prime:
            continue

        # Finalize the shortest path to the current node
        n_prime.add(node)
        step_count += 1 # Increment step count after finalizing a node

        # Yield state after finalizing a node
        yield step_count, set(n_prime), dict(distances), dict(previous) # Yield copies

        # If all nodes are finalized, stop
        if len(n_prime) == len(graph.nodes()):
            break

        # Update distances for neighbors
        for neighbor in graph.neighbors(node):
            # Ensure edge exists and get weight
            if graph.has_edge(node, neighbor):
                weight = graph[node][neighbor].get('weight', 1)
                if weight is None:
                    weight = 1
                try:
                    weight = float(weight)
                except (ValueError, TypeError):
                     print(f"Warning: Invalid weight found for edge ({node}, {neighbor}). Using weight 1.", file=sys.stderr)
                     weight = 1

                new_dist = distances[node] + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = node
                    heapq.heappush(queue, (new_dist, neighbor))

        # Note: Yielding after processing neighbors would show D updates before the next node is finalized.
        # The image shows the state *after* a node is finalized and its neighbors' distances are potentially updated.
        # The current yield points match the image's structure.


    # Final state after loop finishes (all reachable nodes finalized)
    # If the loop finished because the queue is empty but not all nodes are in N' (disconnected graph)
    if len(n_prime) < len(graph.nodes()):
         step_count += 1
         yield step_count, set(n_prime), dict(distances), dict(previous)


# Original dijkstra function (kept for forwarding table calculation)
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start] = 0
    previous = {}
    queue = [(0, start)]

    while queue:
        dist, node = heapq.heappop(queue)

        if dist > distances[node]:
            continue

        for neighbor in graph.neighbors(node):
            weight = graph[node][neighbor].get('weight', 1)
            if weight is None:
                 weight = 1
            try:
                weight = float(weight)
            except (ValueError, TypeError):
                 print(f"Warning: Invalid weight found for edge ({node}, {neighbor}). Using weight 1.", file=sys.stderr)
                 weight = 1

            new_dist = dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(queue, (new_dist, neighbor))

    return distances, previous


# Adapted CreateGraph function to read from a file in token form (space-separated)
def create_network_graph(file_path):
    G = nx.Graph() # Use nx.DiGraph() for directed edges if needed
    error_message = None
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
                # Assuming space-separated tokens based on previous request
                parts = line.strip().split()
                if len(parts) == 3:
                    src, dest, weight_str = parts
                    try:
                        weight = float(weight_str) # Use float for weight
                        G.add_edge(src, dest, weight=weight) # Add edge with weight
                        added_edges += 1
                    except ValueError:
                         print(f"Warning: Skipping line with invalid weight format: {line.strip()}", file=sys.stderr) # Keep warnings in console
                    except Exception as e:
                         print(f"Warning: Error adding edge from line {line.strip()}: {e}", file=sys.stderr) # Keep warnings in console
                else:
                     # Fallback for comma-separated if space-separated fails, or just warn
                     parts_comma = line.strip().split(',')
                     if len(parts_comma) == 3:
                         src, dest, weight_str = parts_comma
                         try:
                             weight = float(weight_str.strip())
                             G.add_edge(src.strip(), dest.strip(), weight=weight)
                             added_edges += 1
                             print(f"Warning: Found comma-separated format in line: {line.strip()}. Expected space-separated.", file=sys.stderr)
                         except ValueError:
                              print(f"Warning: Skipping line with invalid weight format (comma-separated attempt): {line.strip()}", file=sys.stderr)
                         except Exception as e:
                              print(f"Warning: Error adding edge from line (comma-separated attempt) {line.strip()}: {e}", file=sys.stderr)
                     else:
                        print(f"Warning: Skipping malformed line: {line.strip()}. Expected 'src dest weight' or 'src,dest,weight'", file=sys.stderr)


            if added_edges != num_edges:
                 print(f"Warning: Expected {num_edges} edges but added {added_edges}.", file=sys.stderr)

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

    output_text += "--- Generating Final Forwarding Tables ---\n" # Changed title
    for source_node in graph.nodes():
        try:
            # Calculate shortest paths from the current source node to all others
            # networkx's shortest_path uses Dijkstra for weighted graphs by default
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

        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=800, edge_color='k', linewidths=1, font_size=10, ax=ax)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color='red', ax=ax)
        ax.set_title("Network Topology")
        ax.axis('off') # Turn off axes

        return fig, ax, pos, None # Return figure, axes, positions, and no error message
    except Exception as e:
        return None, None, None, f"Error during visualization creation: {e}"


# --- GUI Implementation ---

class NetworkApp:
    def __init__(self, root):
        self.root = root
        root.title("Network Link-State Simulator")
        # Maximizing the window is handled in the __main__ block

        self.graph = None # To store the network graph
        self.pos = None # To store node positions for consistent layout
        self.canvas = None # To store the matplotlib canvas
        self.toolbar = None # To store the matplotlib toolbar
        self.ax = None # To store the matplotlib axes for drawing

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
        output_frame = ttk.LabelFrame(left_pane, text="Output (Graph Info & Tables)") # Simplified title
        output_frame.pack(pady=5, fill="both", expand=True)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=60, height=15, font=('Courier New', 10)) # Use a monospace font for tables
        self.output_text.pack(padx=5, pady=5, fill="both", expand=True)

        # Animation/Step-by-Step Controls Frame
        animation_steps_frame = ttk.LabelFrame(left_pane, text="Algorithm Visualization") # Combined frame
        animation_steps_frame.pack(pady=10, fill="x")

        # Source Node Selection (Used for both Animation and Steps)
        source_frame = ttk.Frame(animation_steps_frame)
        source_frame.pack(pady=2, fill="x")
        ttk.Label(source_frame, text="Source Node:").pack(side="left", padx=5)
        self.source_node_combobox = ttk.Combobox(source_frame, state="disabled", width=10)
        self.source_node_combobox.pack(side="left", padx=5, expand=True, fill="x")

        # Destination Node Selection (Only for Animation)
        dest_frame = ttk.Frame(animation_steps_frame)
        dest_frame.pack(pady=2, fill="x")
        ttk.Label(dest_frame, text="Destination Node:").pack(side="left", padx=5)
        self.dest_node_combobox = ttk.Combobox(dest_frame, state="disabled", width=10)
        self.dest_node_combobox.pack(side="left", padx=5, expand=True, fill="x")

        # Buttons Frame for Animation and Steps
        buttons_frame = ttk.Frame(animation_steps_frame)
        buttons_frame.pack(pady=5)

        # Animate Button (Existing)
        self.animate_button = ttk.Button(buttons_frame, text="Animate Shortest Path", command=self.start_animation, state="disabled")
        self.animate_button.pack(side="left", padx=5)

        # Show Dijkstra Steps Button (New)
        self.show_steps_button = ttk.Button(buttons_frame, text="Show Dijkstra Steps", command=self.show_dijkstra_steps, state="disabled")
        self.show_steps_button.pack(side="left", padx=5)


        # Right Pane: Visualization
        right_pane = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(right_pane, sticky="nsew")

        self.visualization_frame = ttk.LabelFrame(right_pane, text="Network Visualization")
        self.visualization_frame.pack(pady=5, fill="both", expand=True)

        # Placeholder for the matplotlib canvas
        self.plot_container = ttk.Frame(self.visualization_frame)
        self.plot_container.pack(fill="both", expand=True)

        # Animation state variables for smoother movement (Moved here)
        self.animation_marker = None # To store the animation marker artist
        self.animation_path = [] # List of nodes in the shortest path
        self.current_path_segment_index = 0 # Index of the current edge being animated (0 to len(path)-2)
        self.current_segment_progress = 0.0 # Progress along the current edge (0.0 to 1.0)
        self.segment_steps = 20 # Number of small steps to take along each edge for smoother animation
        self.animation_speed = 50 # Milliseconds delay between each small animation step (adjust for overall speed)
        self._animation_job_id = None # To store the ID returned by root.after()


    def browse_file(self):
        """Opens a file dialog to select the input file."""
        file_path = filedialog.askopenfilename(
            initialdir=".", # Start Browse from the current directory
            title="Select Network Input File",
            filetypes=(("Text Files", "*.txt"), ("All files", "*.*"))
        )
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)
            # Disable animation/steps controls until a graph is processed
            self.source_node_combobox['state'] = 'disabled'
            self.dest_node_combobox['state'] = 'disabled'
            self.animate_button['state'] = 'disabled'
            self.show_steps_button['state'] = 'disabled' # Disable new button
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

        # --- Step 1: Create the network graph from file ---
        self.graph, error_message = create_network_graph(file_path)

        if error_message:
            messagebox.showerror("File Reading Error", error_message)
            self.output_text.insert(tk.END, f"Error: {error_message}\n")
            # Disable animation/steps controls on error
            self.source_node_combobox['state'] = 'disabled'
            self.dest_node_combobox['state'] = 'disabled'
            self.animate_button['state'] = 'disabled'
            self.show_steps_button['state'] = 'disabled' # Disable new button
            self.source_node_combobox.set('')
            self.dest_node_combobox.set('')
            self.clear_plot() # Clear any previous plot on error
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

            # --- Step 2: Visualize the network graph ---
            self.clear_plot() # Clear previous plot before drawing new one
            fig, ax, self.pos, viz_error = create_network_visualization(self.graph)

            if viz_error:
                 messagebox.showerror("Visualization Error", viz_error)
                 self.output_text.insert(tk.END, f"Visualization Error: {viz_error}\n")
                 # Disable animation/steps controls on visualization error
                 self.source_node_combobox['state'] = 'disabled'
                 self.dest_node_combobox['state'] = 'disabled'
                 self.animate_button['state'] = 'disabled'
                 self.show_steps_button['state'] = 'disabled' # Disable new button
                 self.source_node_combobox.set('')
                 self.dest_node_combobox.set('')
            elif fig and ax and self.pos:
                self.embed_plot(fig, ax)
                self.output_text.insert(tk.END, "\nNetwork visualization created.\n")
                self.output_text.update()

                # Enable and populate animation/steps controls
                self.source_node_combobox['values'] = nodes
                self.dest_node_combobox['values'] = nodes
                self.source_node_combobox['state'] = 'readonly' # Prevent typing
                self.dest_node_combobox['state'] = 'readonly' # Prevent typing
                self.animate_button['state'] = 'normal'
                self.show_steps_button['state'] = 'normal' # Enable new button

                # --- Step 3: Compute and display FINAL forwarding tables ---
                # Displaying final tables here, step-by-step is a separate action
                all_tables, tables_output_text = generate_forwarding_tables(self.graph)
                self.output_text.insert(tk.END, tables_output_text)
                self.output_text.update()


        else:
            self.output_text.insert(tk.END, "\nFailed to create network graph.\n")
            # Disable animation/steps controls if graph creation failed
            self.source_node_combobox['state'] = 'disabled'
            self.dest_node_combobox['state'] = 'disabled'
            self.animate_button['state'] = 'disabled'
            self.show_steps_button['state'] = 'disabled' # Disable new button
            self.source_node_combobox.set('')
            self.dest_node_combobox.set('')
            self.clear_plot() # Clear any previous plot


    def embed_plot(self, figure, axes):
        """Embeds a matplotlib figure into the plot_container frame."""
        # Clear previous canvas and toolbar if they exist
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()

        # Create a canvas widget for the figure
        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_container)
        self.canvas.draw()

        # Create a toolbar for the canvas (optional, adds interactivity)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_container)
        self.toolbar.update()

        # Pack the canvas and toolbar
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Store axes for potential future use (e.g., animation, steps marker)
        self.ax = axes

    def clear_plot(self):
        """Clears the embedded matplotlib plot and toolbar."""
         # Check if animation is running and stop it if necessary
        if hasattr(self, '_animation_job_id') and self._animation_job_id is not None:
             self.root.after_cancel(self._animation_job_id)
             self._animation_job_id = None
        # Remove animation marker if it exists
        if hasattr(self, 'animation_marker') and self.animation_marker:
            try:
                self.animation_marker.remove()
                self.animation_marker = None
                if self.canvas: # If canvas exists, update plot after removing marker
                    self.canvas.draw_idle()
            except Exception as e:
                 print(f"Error removing animation marker: {e}", file=sys.stderr)

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
        # (Animation logic remains the same as the previous version)
        # This function is quite long, keeping it collapsed for clarity of changes
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
        # (Animation step logic remains the same as the previous version)
        # This function is quite long, keeping it collapsed for clarity of changes

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


    def show_dijkstra_steps(self):
        """
        Runs Dijkstra's algorithm step-by-step for the selected source node
         and displays the intermediate states in the output text area.
        """
        if self.graph is None:
            messagebox.showwarning("Steps Error", "Network graph not processed yet.")
            return

        source_node = self.source_node_combobox.get()

        if not source_node:
            messagebox.showwarning("Steps Error", "Please select a source node.")
            return

        if source_node not in self.graph.nodes():
             messagebox.showwarning("Steps Error", "Selected source node is not in the graph.")
             return

        self.output_text.insert(tk.END, f"\n\n--- Dijkstra Algorithm Steps for Node '{source_node}' ---\n")
        self.output_text.update()

        nodes = sorted(list(self.graph.nodes())) # Get sorted nodes for consistent table columns
        # Adjust column widths to match the image visually
        col_widths = {'step': 5, 'N\'': 10}
        # Dynamically determine width for D(),p() columns based on node names and potential values
        # A fixed width of 15-18 seems reasonable for single-char node names and "inf,node" format
        d_p_col_width = 18 # Adjusted width

        header = f"{'Step':<{col_widths['step']}}{'N\'':<{col_widths['N\'']}}"
        for node in nodes:
             header += f"{f'D({node}),p({node})':<{d_p_col_width}}"
        header += "\n" + "-" * (col_widths['step'] + col_widths['N\''] + len(nodes) * d_p_col_width) + "\n"
        self.output_text.insert(tk.END, header)
        self.output_text.update()


        try:
            for step, n_prime, distances, previous in dijkstra_step_by_step(self.graph, source_node):
                # Format N' as {u,x,y}
                n_prime_str = '{' + ','.join(sorted(list(n_prime))) + '}'
                row = f"{step:<{col_widths['step']}}{n_prime_str:<{col_widths['N\'']}}"

                for node in nodes:
                    dist_val = distances.get(node, float('inf'))
                    pred_val = previous.get(node, '-') # Use '-' if no predecessor yet

                    # Format distance: .1f for finite, 'inf' for infinity
                    dist_str = f"{dist_val:.1f}" if dist_val != float('inf') else "inf"
                    cell_content = f"{dist_str},{pred_val}"
                    row += f"{cell_content:<{d_p_col_width}}"

                self.output_text.insert(tk.END, row + "\n")
                self.output_text.see(tk.END) # Auto-scroll to the bottom
                self.output_text.update()
                # Add a small delay to see steps appear gradually (optional)
                # self.root.after(200) # Uncomment and adjust delay if needed

            self.output_text.insert(tk.END, "--- End of Steps ---\n")
            self.output_text.update()


        except Exception as e:
            messagebox.showerror("Steps Error", f"An error occurred during step-by-step execution: {e}")
            self.output_text.insert(tk.END, f"\nError during step-by-step execution: {e}\n")
            self.output_text.update()



# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed') # Attempt to maximize the window using 'zoomed' state (Windows)
    # For Linux/X11, you might need:
    # root.attributes('-zoomed', True)
    # Or set geometry to screen size:
    # screen_width = root.winfo_screenwidth()
    # screen_height = root.winfo_screenheight()
    # root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.update_idletasks() # Process geometry requests

    app = NetworkApp(root)
    root.mainloop()
