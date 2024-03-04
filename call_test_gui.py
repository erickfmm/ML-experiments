import os
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from chlorophyll import CodeView
from pygments.lexers import PythonLexer
from tkterm import Terminal

class CodeRunnerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Code Runner")

        # Menu
        menu_bar = tk.Menu(root)
        root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder", command=self.open_folder)
        self.folder_path = "./test/"
        # Main frame with three columns
        main_frame = tk.Frame(root)
        main_frame.pack(expand=False, fill="both", padx=10, pady=10)

        # Column 1: Buttons
        self.button_frame = tk.Frame(main_frame)
        self.button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        # Column 2: Code Viewer
        #self.code_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=40, height=20)
        #self.code_text.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.code_text = CodeView(main_frame, lexer=PythonLexer, color_scheme="monokai")
        self.code_text.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Column 3: Terminal Output
        self.terminal_text = Terminal(main_frame) #scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=40, height=20)
        self.terminal_text.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        # Run and Stop Buttons
        self.run_button = tk.Button(root, text="Run Code", command=self.run_code)
        self.run_button.pack(side="left", padx=10)
        #self.stop_button = tk.Button(root, text="Stop Running", command=self.stop_running)
        #self.stop_button.pack(side="left", padx=10)

        

        # Initialize buttons in the first column
        self.load_buttons()

    def open_folder(self):
        folder_path = filedialog.askdirectory(title="Open Folder", initialdir="./test/")
        if folder_path:
            self.folder_path = folder_path
            self.load_buttons()

    def load_buttons(self):
        # Clear existing buttons
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        # Create buttons for each .py file in the folder
        if hasattr(self, 'folder_path'):
            list_of_files = os.listdir(self.folder_path)
            list_of_files.sort()
            py_files = [f for f in list_of_files if f.endswith(".py")]
            for file in py_files:
                btn = tk.Button(self.button_frame, text=file, command=lambda f=file: self.load_code(f))
                btn.pack(side="top", pady=5)

    def load_code(self, file_name):
        self.file_path = os.path.join(self.folder_path, file_name)
        with open(self.file_path, 'r') as file:
            code_content = file.read()

        # Apply syntax highlighting using Pygments
        self.code_text.delete(1.0, tk.END)
        self.highlight_code(code_content)

    def highlight_code(self, code_content):
        # Use Pygments to apply syntax highlighting
        # Use IDLE's colorizer to apply syntax highlighting
        self.code_text.insert(tk.END, code_content)

    def run_code(self):
        try:
            self.terminal_text.run_command(f'"{sys.executable}" "{self.file_path}"')
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CodeRunnerApp(root)
    root.mainloop()
