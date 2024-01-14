import os
import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import messagebox
import subprocess
import threading
import queue
import pygments.lexers
from chlorophyll import CodeView
from pygments.lexers import PythonLexer

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

        # Main frame with three columns
        main_frame = tk.Frame(root)
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Column 1: Buttons
        self.button_frame = tk.Frame(main_frame)
        self.button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        # Column 2: Code Viewer
        #self.code_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=40, height=20)
        #self.code_text.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.code_text = CodeView(main_frame, lexer=PythonLexer, color_scheme="monokai")
        self.code_text.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Column 3: Terminal Output
        self.terminal_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=40, height=20)
        self.terminal_text.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        # Run and Stop Buttons
        self.run_button = tk.Button(root, text="Run Code", command=self.run_code)
        self.run_button.pack(side="left", padx=10)
        self.stop_button = tk.Button(root, text="Stop Running", command=self.stop_running, state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=10)

        # Initialize buttons in the first column
        self.load_buttons()

    def open_folder(self):
        folder_path = filedialog.askdirectory(title="Open Folder")
        if folder_path:
            self.folder_path = folder_path
            self.load_buttons()

    def load_buttons(self):
        # Clear existing buttons
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        # Create buttons for each .py file in the folder
        if hasattr(self, 'folder_path'):
            py_files = [f for f in os.listdir(self.folder_path) if f.endswith(".py")]
            for file in py_files:
                btn = tk.Button(self.button_frame, text=file, command=lambda f=file: self.load_code(f))
                btn.pack(side="top", pady=5)

    def load_code(self, file_name):
        file_path = os.path.join(self.folder_path, file_name)
        with open(file_path, 'r') as file:
            code_content = file.read()

        # Apply syntax highlighting using Pygments
        self.code_text.delete(1.0, tk.END)
        self.highlight_code(code_content)

    def highlight_code(self, code_content):
        # Use Pygments to apply syntax highlighting
        # Use IDLE's colorizer to apply syntax highlighting
        self.code_text.insert(tk.END, code_content)

    def run_code(self):
        code_content = self.code_text.get(1.0, tk.END)
        self.terminal_text.delete(1.0, tk.END)

        try:
            self.process = subprocess.Popen(["python", "-c", code_content], 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.PIPE, 
                                            text=True, 
                                            shell=True, 
                                            bufsize=1,
                                            universal_newlines=True)

            self.stop_button.config(state=tk.NORMAL)

            # Start a separate thread to read stdout and stderr in real-time
            self.output_thread = threading.Thread(target=self.read_output, daemon=True)
            self.output_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def read_output(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            self.terminal_text.insert(tk.END, line)
            self.terminal_text.yview(tk.END)

        # Wait for the process to finish and get the remaining output
        remaining_output, _ = self.process.communicate()
        self.terminal_text.insert(tk.END, remaining_output)

        # Enable the "Run Code" button and disable the "Stop Running" button
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def stop_running(self):
        # Terminate the running process
        if hasattr(self, 'process') and self.process.poll() is None:
            self.process.terminate()

if __name__ == "__main__":
    root = tk.Tk()
    app = CodeRunnerApp(root)
    root.mainloop()
