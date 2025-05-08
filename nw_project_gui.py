import networkx as nx
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading # To run visualization without freezing the GUI

# --- Existing Network Logic Functions (Copied from previous immersive) ---

def create_network_graph(file_path):
    """
    Reads network topology from a file and creates a networkx graph.
    Reads the number of nodes and edges from the first line.
    Reads edges (src, dest, weight) from subsequent lines.

    Args:
        file_path (str): The path to the input file.

    Returns:
        nx.Graph: The created graph object or None if file not found or parsing error occurs.
    """
    G = nx.Graph() # Use nx.DiGraph() for directed edges if needed
    try:
        with open(file_path, 'r') as f:
            # Read number of nodes and edges from the first line
            header = f.readline().split()
            if len(header) != 2:
                 # Use return value for error in GUI context instead of stderr print
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
        for dest in sorted(table.keys()):
             output_text += f"{dest:<11}| {table[dest]}\n"

    return all_forwarding_tables, output_text


def visualize_network(graph):
    """
    Visualizes the network graph using matplotlib.
    Uses plt.show(block=False) to avoid freezing the GUI.

    Args:
        graph (nx.Graph): The network graph.
    """
    if graph is None:
        print("Cannot visualize: Graph is None.", file=sys.stderr)
        return

    print("\nVisualizing the network graph...") # Keep console print for debugging
    try:
        # Create a new figure for each visualization call
        plt.figure()
        pos = nx.spring_layout(graph)

        nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=800, edge_color='k', linewidths=1, font_size=15)

        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

        plt.title("Network Topology")
        # Use block=False to allow the GUI to remain responsive
        plt.show(block=False)
    except Exception as e:
        print(f"Error during visualization: {e}", file=sys.stderr) # Keep console print for debugging


# --- GUI Implementation ---

class NetworkApp:
    def __init__(self, root):
        self.root = root
        root.title("Network Link-State Simulator")

        self.graph = None # To store the network graph

        # --- GUI Elements ---

        # File Selection Frame
        file_frame = tk.LabelFrame(root, text="Input File")
        file_frame.pack(pady=10, padx=10, fill="x")

        self.file_path_entry = tk.Entry(file_frame, width=50)
        self.file_path_entry.pack(side="left", padx=5, pady=5, expand=True, fill="x")

        browse_button = tk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side="left", padx=5, pady=5)

        # Process Button
        process_button = tk.Button(root, text="Process Network", command=self.process_network)
        process_button.pack(pady=5, padx=10)

        # Output Area
        output_frame = tk.LabelFrame(root, text="Output (Graph Info & Forwarding Tables)")
        output_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=80, height=20)
        self.output_text.pack(padx=5, pady=5, fill="both", expand=True)

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
            return

        if self.graph:
            self.output_text.insert(tk.END, "\nGraph created successfully!\n")
            self.output_text.insert(tk.END, f"Number of nodes: {self.graph.number_of_nodes()}\n")
            self.output_text.insert(tk.END, f"Number of edges: {self.graph.number_of_edges()}\n")
            self.output_text.insert(tk.END, f"Nodes: {list(self.graph.nodes())}\n")
            # Edges output can be verbose, maybe print only if graph is small or add an option
            # self.output_text.insert(tk.END, f"Edges (with weights): {list(self.graph.edges(data=True))}\n")
            self.output_text.update()

            # --- Step 2: Generate and display forwarding tables ---
            all_tables, tables_output_text = generate_forwarding_tables(self.graph)
            self.output_text.insert(tk.END, tables_output_text)
            self.output_text.update()

            # --- Step 3: Visualize the network graph ---
            # Run visualization in a separate thread to keep the GUI responsive
            # Matplotlib plots can sometimes block the main thread
            viz_thread = threading.Thread(target=visualize_network, args=(self.graph,))
            viz_thread.start()

        else:
            self.output_text.insert(tk.END, "\nFailed to create network graph.\n")


# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkApp(root)
    root.mainloop()
