import os
import sys
import io
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


class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, string):
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, string)
        self.widget.configure(state="disabled")


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
        self.stop_button = tk.Button(root, text="Stop Running", command=self.stop_running)
        self.stop_button.pack(side="left", padx=10)

        # Create a queue for communication between threads
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Redirect sys.stdout, sys.stderr, and sys.stdin to the terminal
        sys.stdout = self.TerminalOutput(self.output_queue)
        sys.stderr = self.TerminalOutput(self.output_queue)
        sys.stdin = self.TerminalInput(self.input_queue)

        # Start a separate thread for the terminal emulator
        self.terminal_thread = threading.Thread(target=self.run_terminal_emulator, daemon=True)
        self.terminal_thread.start()

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
        #code_content = self.code_text.get(1.0, tk.END)
        self.terminal_text.delete(1.0, tk.END)

        try:
            # Redirect sys.stdout, sys.stderr, and sys.stdin to the terminal
            sys.stdout = self.TerminalOutput(self.output_queue)
            sys.stderr = self.TerminalOutput(self.output_queue)
            sys.stdin = self.TerminalInput(self.input_queue)

            # Start a separate thread for running the code and capturing output
            self.subprocess_thread = threading.Thread(
                target=self.run_subprocess,
                args=([sys.executable, self.file_path],)
            )
            self.subprocess_thread.start()
            

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def run_terminal_emulator(self):
        # Run a subprocess to emulate a terminal
        self.process = subprocess.Popen(
            [os.environ.get("SHELL", "/bin/bash")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Start two threads for handling stdout and stderr
        stdout_thread = threading.Thread(target=self.read_output, args=(self.process.stdout,))
        stderr_thread = threading.Thread(target=self.read_output, args=(self.process.stderr,))
        stdout_thread.start()
        stderr_thread.start()

        # Main loop for the terminal emulator
        while True:
            # Check if there is any input from the input queue
            try:
                input_data = self.input_queue.get_nowait()
                self.process.stdin.write(input_data + '\n')
                self.process.stdin.flush()
            except queue.Empty:
                pass

            # Check if there is any output from the output queue
            try:
                output_data = self.output_queue.get_nowait()
                self.terminal_text.insert(tk.END, output_data)
                self.terminal_text.yview(tk.END)
            except queue.Empty:
                pass

            # Check if the subprocess has terminated
            return_code = self.process.poll()
            if return_code is not None:
                break

        # Cleanup: Restore original sys.stdin, sys.stdout, and sys.stderr
        #sys.stdin = sys.__stdin__
        #sys.stdout = sys.__stdout__
        #sys.stderr = sys.__stderr__

    def run_subprocess(self, command):
        try:
            # Run the code using subprocess.Popen and pass the redirected streams
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True
            )

            # Start two threads for handling stdout and stderr
            stdout_thread = threading.Thread(target=self.read_output, args=(process.stdout,))
            stderr_thread = threading.Thread(target=self.read_output, args=(process.stderr,))
            stdout_thread.start()
            stderr_thread.start()

            # Feed the subprocess stdin with an empty string
            process.communicate("")

            # Wait for both threads to finish
            stdout_thread.join()
            stderr_thread.join()

        finally:
            # Cleanup: Restore original sys.stdin, sys.stdout, and sys.stderr
            sys.stdin = sys.__stdin__
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            # Notify the main thread that the subprocess has finished
            self.root.after(0, self.subprocess_finished)

    def read_output(self, stream):
        for line in iter(stream.readline, ""):
            self.output_queue.put(line)

    def subprocess_finished(self):
        # You can perform any cleanup or additional actions here
        pass

    def stop_running(self):
        # Terminate the running process
        if hasattr(self, 'file_process') and self.file_process.poll() is None:
            self.file_process.terminate()

    class TerminalOutput(io.TextIOBase):
        def __init__(self, output_queue):
            self.output_queue = output_queue

        def write(self, text):
            self.output_queue.put(text)

        def flush(self):
            pass

        def fileno(self):
            return 1

    class TerminalInput(io.TextIOBase):
        def __init__(self, input_queue):
            self.input_queue = input_queue

        def readline(self):
            return self.input_queue.get()
        def fileno(self):
            return 0

if __name__ == "__main__":
    root = tk.Tk()
    app = CodeRunnerApp(root)
    root.mainloop()
