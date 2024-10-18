import tkinter as tk
from tkinter import messagebox, simpledialog

class Node:
    def __init__(self, name, x, y, canvas):
        self.name = name
        self.x = x
        self.y = y
        self.canvas = canvas
        self.radius = 15
        self.circle = None
        self.text = None
        self.draw()
        self.bind()

    def draw(self):
        r = self.radius
        self.circle = self.canvas.create_oval(
            self.x - r, self.y - r, self.x + r, self.y + r,
            fill="lightblue", outline="black", width=2, tags="node"
        )
        self.text = self.canvas.create_text(
            self.x, self.y, text=self.name, font=("Arial", 12, "bold"), tags="node"
        )

    def bind(self):
        # Left-click for selecting
        self.canvas.tag_bind(self.circle, "<Button-1>", self.on_press)
        self.canvas.tag_bind(self.text, "<Button-1>", self.on_press)
        # Left-click drag for moving
        self.canvas.tag_bind(self.circle, "<B1-Motion>", self.on_move)
        self.canvas.tag_bind(self.text, "<B1-Motion>", self.on_move)
        # Right-click for context menu (Delete Node)
        self.canvas.tag_bind(self.circle, "<Button-3>", self.show_context_menu)
        self.canvas.tag_bind(self.text, "<Button-3>", self.show_context_menu)

    def on_press(self, event):
        self.lastx = event.x
        self.lasty = event.y

    def on_move(self, event):
        dx = event.x - self.lastx
        dy = event.y - self.lasty
        self.canvas.move(self.circle, dx, dy)
        self.canvas.move(self.text, dx, dy)
        self.x += dx
        self.y += dy
        self.lastx = event.x
        self.lasty = event.y
        self.canvas.event_generate("<<NodeMoved>>", when="tail")

    def show_context_menu(self, event):
        menu = tk.Menu(self.canvas, tearoff=0)
        menu.add_command(label="Delete Node", command=self.delete_node)
        menu.tk_popup(event.x_root, event.y_root)

    def delete_node(self):
        app = self.canvas.master.app
        app.delete_node(self)

class Connection:
    def __init__(self, node1, node2, canvas):
        self.node1 = node1
        self.node2 = node2
        self.canvas = canvas
        self.line = None
        self.connected = True
        self.draw()
        self.bind()

    def draw(self):
        color = "green" if self.connected else "red"
        self.line = self.canvas.create_line(
            self.node1.x, self.node1.y, self.node2.x, self.node2.y,
            fill=color, width=4, tags="connection"
        )

    def bind(self):
        # Left-click to toggle connection
        self.canvas.tag_bind(self.line, "<Button-1>", self.toggle_connection)
        # Right-click for context menu to delete connection
        self.canvas.tag_bind(self.line, "<Button-3>", self.show_context_menu)

    def toggle_connection(self, event):
        self.connected = not self.connected
        new_color = "green" if self.connected else "red"
        self.canvas.itemconfig(self.line, fill=new_color)

    def show_context_menu(self, event):
        menu = tk.Menu(self.canvas, tearoff=0)
        menu.add_command(label="Delete Connection", command=self.delete_connection)
        menu.tk_popup(event.x_root, event.y_root)

    def delete_connection(self):
        app = self.canvas.master.app
        app.delete_connection(self)

    def update_position(self):
        self.canvas.coords(
            self.line, self.node1.x, self.node1.y, self.node2.x, self.node2.y
        )
        new_color = "green" if self.connected else "red"
        self.canvas.itemconfig(self.line, fill=new_color)

class TextLabel:
    def __init__(self, text, x, y, canvas):
        self.text = text
        self.x = x
        self.y = y
        self.canvas = canvas
        self.item = None
        self.draw()
        self.bind()

    def draw(self):
        self.item = self.canvas.create_text(
            self.x, self.y, text=self.text, font=("Arial", 14),
            fill="black", tags="textlabel"
        )

    def bind(self):
        # Left-click drag to move
        self.canvas.tag_bind(self.item, "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind(self.item, "<B1-Motion>", self.on_move)
        # Right-click for context menu (Edit/Delete)
        self.canvas.tag_bind(self.item, "<Button-3>", self.show_context_menu)

    def on_press(self, event):
        self.lastx = event.x
        self.lasty = event.y

    def on_move(self, event):
        dx = event.x - self.lastx
        dy = event.y - self.lasty
        self.canvas.move(self.item, dx, dy)
        self.x += dx
        self.y += dy
        self.lastx = event.x
        self.lasty = event.y

    def show_context_menu(self, event):
        menu = tk.Menu(self.canvas, tearoff=0)
        menu.add_command(label="Edit Text", command=self.edit_text)
        menu.add_command(label="Delete Text", command=self.delete_text)
        menu.tk_popup(event.x_root, event.y_root)

    def edit_text(self):
        app = self.canvas.master.app
        new_text = simpledialog.askstring(
            "Edit Text", "Enter new text:", initialvalue=self.text, parent=self.canvas.master
        )
        if new_text:
            self.text = new_text
            self.canvas.itemconfig(self.item, text=self.text)

    def delete_text(self):
        app = self.canvas.master.app
        app.delete_text_label(self)

class CircuitSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Circuit Simulator")
        self.nodes = {}
        self.connections = []
        self.text_labels = []
        self.selected_nodes = []
        self.adding_text = False
        self.adding_node = False
        self.scale = 1.0  # Initial scale
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.current_node_name = ""
        self.setup_ui()
        self.create_default_nodes()
        self.canvas.bind("<<NodeMoved>>", self.update_connections)
        self.bind_zoom()
        self.bind_panning()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def setup_ui(self):
        # Create the canvas
        self.canvas = tk.Canvas(self.root, width=1000, height=700, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Control panel
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10)

        self.connect_button = tk.Button(
            self.control_frame, text="Connect Nodes", command=self.connect_nodes
        )
        self.connect_button.grid(row=0, column=0, padx=5)

        self.disconnect_button = tk.Button(
            self.control_frame, text="Disconnect Nodes", command=self.disconnect_nodes
        )
        self.disconnect_button.grid(row=0, column=1, padx=5)

        self.add_text_button = tk.Button(
            self.control_frame, text="Add Text", command=self.start_add_text
        )
        self.add_text_button.grid(row=0, column=2, padx=5)

        self.add_node_button = tk.Button(
            self.control_frame, text="Add Node", command=self.start_add_node
        )
        self.add_node_button.grid(row=0, column=3, padx=5)

        self.delete_node_button = tk.Button(
            self.control_frame, text="Delete Node", command=self.delete_selected_nodes
        )
        self.delete_node_button.grid(row=0, column=4, padx=5)

        self.reset_button = tk.Button(
            self.control_frame, text="Reset Connections", command=self.reset_connections
        )
        self.reset_button.grid(row=0, column=5, padx=5)

        # Instructions
        instructions = (
            "Instructions:\n"
            "1. Click on two nodes to select them for connection or disconnection.\n"
            "2. Use 'Connect Nodes' to create a connection (green).\n"
            "3. Use 'Disconnect Nodes' to remove a connection (red).\n"
            "4. Use 'Add Text' to place a draggable text label on the canvas.\n"
            "5. Right-click on a connection line to delete it.\n"
            "6. Right-click on a text label to edit or delete it.\n"
            "7. Drag nodes or text labels to reposition them.\n"
            "8. Use the mouse wheel to zoom in and out.\n"
            "9. Click and hold the left mouse button on empty space to pan around the canvas.\n"
            "10. Use 'Add Node' to create new nodes by clicking on the canvas.\n"
            "11. Select nodes and use 'Delete Node' to remove them along with their connections."
        )
        self.instructions = tk.Label(
            self.root, text=instructions, justify="left", fg="blue"
        )
        self.instructions.pack(pady=10)

        # Assign app to root for access in classes
        self.root.app = self

    def create_default_nodes(self):
        default_nodes = {
            'A': (200, 200),
            'B': (800, 200),
            'C': (200, 500),
            'D': (800, 500)
        }
        for name, (x, y) in default_nodes.items():
            node = Node(name, x, y, self.canvas)
            self.nodes[name] = node
            self.canvas.tag_bind(node.circle, "<Button-1>", lambda event, n=node: self.select_node(n))
            self.canvas.tag_bind(node.text, "<Button-1>", lambda event, n=node: self.select_node(n))

    def select_node(self, node):
        if self.adding_node or self.adding_text:
            # If in adding node or text mode, ignore selection
            return
        if node in self.selected_nodes:
            self.selected_nodes.remove(node)
            self.canvas.itemconfig(node.circle, fill="lightblue")
        else:
            if len(self.selected_nodes) < 2:
                self.selected_nodes.append(node)
                self.canvas.itemconfig(node.circle, fill="lightgreen")
            else:
                messagebox.showinfo("Selection Limit", "You can only select two nodes at a time.")
        # Highlight selected nodes
        for n in self.nodes.values():
            if n in self.selected_nodes:
                self.canvas.itemconfig(n.circle, fill="lightgreen")
            else:
                self.canvas.itemconfig(n.circle, fill="lightblue")

    def connect_nodes(self):
        if len(self.selected_nodes) != 2:
            messagebox.showerror("Selection Error", "Please select exactly two nodes to connect.")
            return
        node1, node2 = self.selected_nodes
        # Check if connection already exists
        for conn in self.connections:
            if (conn.node1 == node1 and conn.node2 == node2) or (conn.node1 == node2 and conn.node2 == node1):
                messagebox.showinfo("Connection Exists", "These nodes are already connected.")
                self.selected_nodes = []
                self.update_node_colors()
                return
        # Create new connection
        connection = Connection(node1, node2, self.canvas)
        self.connections.append(connection)
        self.selected_nodes = []
        self.update_node_colors()

    def disconnect_nodes(self):
        if len(self.selected_nodes) != 2:
            messagebox.showerror("Selection Error", "Please select exactly two nodes to disconnect.")
            return
        node1, node2 = self.selected_nodes
        # Find existing connection
        for conn in self.connections:
            if (conn.node1 == node1 and conn.node2 == node2) or (conn.node1 == node2 and conn.node2 == node1):
                if not conn.connected:
                    messagebox.showinfo("Already Disconnected", "These nodes are already disconnected.")
                    self.selected_nodes = []
                    self.update_node_colors()
                    return
                conn.connected = False
                self.canvas.itemconfig(conn.line, fill="red")
                self.selected_nodes = []
                self.update_node_colors()
                return
        messagebox.showinfo("No Connection", "These nodes are not connected.")

    def reset_connections(self):
        for conn in self.connections:
            conn.connected = True
            self.canvas.itemconfig(conn.line, fill="green")

    def update_node_colors(self):
        for node in self.nodes.values():
            if node in self.selected_nodes:
                self.canvas.itemconfig(node.circle, fill="lightgreen")
            else:
                self.canvas.itemconfig(node.circle, fill="lightblue")

    def update_connections(self, event=None):
        for conn in self.connections:
            conn.update_position()

    def start_add_text(self):
        self.adding_text = True
        self.adding_node = False
        self.canvas.config(cursor="tcross")
        self.instructions.config(text=(
            "Add Text Mode:\n"
            "Click on the canvas where you want to place the text."
        ))

    def start_add_node(self):
        self.adding_node = True
        self.adding_text = False
        self.canvas.config(cursor="cross")
        node_name = simpledialog.askstring("Add Node", "Enter node name:", parent=self.root)
        if node_name:
            if node_name in self.nodes:
                messagebox.showerror("Duplicate Node", f"Node '{node_name}' already exists.")
                self.adding_node = False
                self.canvas.config(cursor="")
                self.instructions.config(text=(
                    "Instructions:\n"
                    "1. Click on two nodes to select them for connection or disconnection.\n"
                    "2. Use 'Connect Nodes' to create a connection (green).\n"
                    "3. Use 'Disconnect Nodes' to remove a connection (red).\n"
                    "4. Use 'Add Text' to place a draggable text label on the canvas.\n"
                    "5. Right-click on a connection line to delete it.\n"
                    "6. Right-click on a text label to edit or delete it.\n"
                    "7. Drag nodes or text labels to reposition them.\n"
                    "8. Use the mouse wheel to zoom in and out.\n"
                    "9. Click and hold the left mouse button on empty space to pan around the canvas.\n"
                    "10. Use 'Add Node' to create new nodes by clicking on the canvas.\n"
                    "11. Select nodes and use 'Delete Node' to remove them along with their connections."
                ))
            else:
                self.current_node_name = node_name
                self.instructions.config(text=(
                    "Add Node Mode:\n"
                    "Click on the canvas where you want to place the node."
                ))
        else:
            self.adding_node = False
            self.canvas.config(cursor="")
            self.instructions.config(text=(
                "Instructions:\n"
                "1. Click on two nodes to select them for connection or disconnection.\n"
                "2. Use 'Connect Nodes' to create a connection (green).\n"
                "3. Use 'Disconnect Nodes' to remove a connection (red).\n"
                "4. Use 'Add Text' to place a draggable text label on the canvas.\n"
                "5. Right-click on a connection line to delete it.\n"
                "6. Right-click on a text label to edit or delete it.\n"
                "7. Drag nodes or text labels to reposition them.\n"
                "8. Use the mouse wheel to zoom in and out.\n"
                "9. Click and hold the left mouse button on empty space to pan around the canvas.\n"
                "10. Use 'Add Node' to create new nodes by clicking on the canvas.\n"
                "11. Select nodes and use 'Delete Node' to remove them along with their connections."
            ))

    def on_canvas_click(self, event):
        if self.adding_text:
            text = simpledialog.askstring("Input", "Enter text to display:", parent=self.root)
            if text:
                # Adjust for current scale
                x = self.canvas.canvasx(event.x)
                y = self.canvas.canvasy(event.y)
                text_label = TextLabel(text, x, y, self.canvas)
                self.text_labels.append(text_label)
            self.adding_text = False
            self.canvas.config(cursor="")
            self.instructions.config(text=(
                "Instructions:\n"
                "1. Click on two nodes to select them for connection or disconnection.\n"
                "2. Use 'Connect Nodes' to create a connection (green).\n"
                "3. Use 'Disconnect Nodes' to remove a connection (red).\n"
                "4. Use 'Add Text' to place a draggable text label on the canvas.\n"
                "5. Right-click on a connection line to delete it.\n"
                "6. Right-click on a text label to edit or delete it.\n"
                "7. Drag nodes or text labels to reposition them.\n"
                "8. Use the mouse wheel to zoom in and out.\n"
                "9. Click and hold the left mouse button on empty space to pan around the canvas.\n"
                "10. Use 'Add Node' to create new nodes by clicking on the canvas.\n"
                "11. Select nodes and use 'Delete Node' to remove them along with their connections."
            ))
        elif self.adding_node:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            node = Node(self.current_node_name, x, y, self.canvas)
            self.nodes[self.current_node_name] = node
            self.canvas.tag_bind(node.circle, "<Button-1>", lambda event, n=node: self.select_node(n))
            self.canvas.tag_bind(node.text, "<Button-1>", lambda event, n=node: self.select_node(n))
            self.adding_node = False
            self.canvas.config(cursor="")
            self.instructions.config(text=(
                "Instructions:\n"
                "1. Click on two nodes to select them for connection or disconnection.\n"
                "2. Use 'Connect Nodes' to create a connection (green).\n"
                "3. Use 'Disconnect Nodes' to remove a connection (red).\n"
                "4. Use 'Add Text' to place a draggable text label on the canvas.\n"
                "5. Right-click on a connection line to delete it.\n"
                "6. Right-click on a text label to edit or delete it.\n"
                "7. Drag nodes or text labels to reposition them.\n"
                "8. Use the mouse wheel to zoom in and out.\n"
                "9. Click and hold the left mouse button on empty space to pan around the canvas.\n"
                "10. Use 'Add Node' to create new nodes by clicking on the canvas.\n"
                "11. Select nodes and use 'Delete Node' to remove them along with their connections."
            ))

    def delete_selected_nodes(self):
        if not self.selected_nodes:
            messagebox.showinfo("No Selection", "No nodes selected to delete.")
            return
        confirm = messagebox.askyesno("Delete Nodes", "Are you sure you want to delete the selected node(s) and their connections?")
        if confirm:
            for node in self.selected_nodes.copy():
                self.delete_node(node)
            self.selected_nodes = []
            self.update_node_colors()

    def delete_node(self, node):
        # Remove all connections associated with this node
        connections_to_remove = [conn for conn in self.connections if conn.node1 == node or conn.node2 == node]
        for conn in connections_to_remove:
            self.canvas.delete(conn.line)
            self.connections.remove(conn)
        # Remove the node's visual elements
        self.canvas.delete(node.circle)
        self.canvas.delete(node.text)
        # Remove from the nodes dictionary
        del self.nodes[node.name]
        # If the node was selected, remove it from selection
        if node in self.selected_nodes:
            self.selected_nodes.remove(node)

    def delete_connection(self, connection):
        self.canvas.delete(connection.line)
        self.connections.remove(connection)

    def add_text_label(self, text, x, y):
        text_label = TextLabel(text, x, y, self.canvas)
        self.text_labels.append(text_label)

    def delete_text_label(self, text_label):
        self.canvas.delete(text_label.item)
        self.text_labels.remove(text_label)

    def bind_zoom(self):
        # Bind mouse wheel for zooming
        # Windows and MacOS have different event names for mouse wheel
        self.canvas.bind("<MouseWheel>", self.zoom)  # Windows and MacOS
        self.canvas.bind("<Button-4>", self.zoom)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.zoom)    # Linux scroll down

    def zoom(self, event):
        # Determine the zoom direction
        if event.num == 4 or event.delta > 0:
            zoom_factor = 1.1
        elif event.num == 5 or event.delta < 0:
            zoom_factor = 0.9
        else:
            return

        # Limit the scale
        new_scale = self.scale * zoom_factor
        if new_scale < 0.5 or new_scale > 3.0:
            return  # Prevent scaling too much

        self.scale = new_scale

        # Get the mouse position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # Scale all items
        self.canvas.scale("all", x, y, zoom_factor, zoom_factor)

        # Update all node and text positions
        for node in self.nodes.values():
            node.x = x + (node.x - x) * zoom_factor
            node.y = y + (node.y - y) * zoom_factor

        for text_label in self.text_labels:
            text_label.x = x + (text_label.x - x) * zoom_factor
            text_label.y = y + (text_label.y - y) * zoom_factor

        # Update connections positions
        self.update_connections()

    def bind_panning(self):
        # Bind left mouse button for panning on empty canvas
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan)

    def start_pan(self, event):
        # Check if click is on empty space and not in Add Node/Text mode
        if self.adding_node or self.adding_text:
            self.panning = False
            return

        clicked_items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
        tags = [self.canvas.gettags(item) for item in clicked_items]
        flat_tags = [tag for sublist in tags for tag in sublist]
        if "node" in flat_tags or "connection" in flat_tags or "textlabel" in flat_tags:
            # Clicked on a node, connection, or text label; do not pan
            self.panning = False
        else:
            # Clicked on empty space; start panning
            self.panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.canvas.config(cursor="fleur")  # Change cursor to indicate panning
            self.canvas.scan_mark(event.x, event.y)  # Mark the starting point for panning

    def do_pan(self, event):
        if not getattr(self, 'panning', False):
            return
        # Perform the panning
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def end_pan(self, event):
        if self.panning:
            self.panning = False
            self.canvas.config(cursor="")  # Reset cursor

def main():
    root = tk.Tk()
    app = CircuitSimulator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
