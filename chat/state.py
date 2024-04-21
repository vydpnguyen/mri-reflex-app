import reflex as rx
import google.generativeai as genai
from pathlib import Path
import paramiko
from paramiko import SSHClient
from scp import SCPClient

# Define the API key directly in the script
API_KEY = "AIzaSyDGY_bjuq_LwQOfbo1rhFuvCLstU9Fv51E"
genai.configure(api_key=API_KEY)

class QA(rx.Base):
    """A question and answer pair."""
    question: str
    answer: str

DEFAULT_CHATS = {
    "Intros": [],
}


class State(rx.State):
    def upload_to_server_via_ssh(self, file_path):
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(self.ssh_host, self.ssh_port, self.ssh_username, self.ssh_password)
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(file_path, self.remote_image_path)
        ssh.close()

    def execute_command_via_ssh(self, command):
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(self.ssh_host, self.ssh_port, self.ssh_username, self.ssh_password)

        stdin, stdout, stderr = ssh.exec_command(command)
        result = stdout.read()

        ssh.close()
        return result.decode('utf-8')

    img = []

    ssh_host = '100.82.44.139'
    ssh_port = 22
    ssh_username = 'ubuntu'
    ssh_password = '1230'

    remote_model_path = '/home/health_app/train/training.py'
    remote_image_path = '/home/health_app/test/Testing'

    """Handling image upload here"""

    img: list[str]
    async def handle_upload(self, files: list[rx.UploadFile]):
        for file in files:
            file_data = await file.read()
            file_path = Path('./uploaded_files/') / file.filename
            file_path.write_bytes(file_data)

            # Append the file path to the list for displaying in the UI
            self.img.append(str(file_path))

            # Send the file to the server
            self.upload_to_server_via_ssh(file_path)

            # Execute the model processing script on the server via SSH
            result = self.execute_command_via_ssh(f'python {self.remote_model_path} {self.remote_image_path}{file.filename}')

            # Process the result
            print(f"Model result for {file.filename}: {result}")

    """The app state."""
    chats: dict[str, list[QA]] = DEFAULT_CHATS
    current_chat = "Intros"
    processing: bool = False
    new_chat_name: str = ""

    def create_chat(self):
        """Create a new chat."""
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat."""
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles."""
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        question = form_data["question"]
        if question == "":
            return

        async for value in self.generate_response(question):
            yield value

    async def generate_response(self, question: str):
        """Process the question using Google's generative AI API."""
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)
        self.processing = True
        yield

        full_prompt = ("You are MIRAI, a Medical AI assistant whose sole focus is on brain tumors, its symptoms, " +
                       "and courses of action that can help cure this disease. You cannot answer questions that " +
                       "are not related to brain tumors, its symptoms, or its cures." + question)
    
        # Instantiate the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate the content
        response = model.generate_content(full_prompt)
    
        
        # Retrieve the text from the response
        answer_text = response.text if response.text else ""
        self.chats[self.current_chat][-1].answer += answer_text
        self.chats = self.chats
        yield

        self.processing = False
