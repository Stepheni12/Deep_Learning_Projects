import tkinter as tk
import pyautogui
from PIL import ImageGrab, Image, ImageTk, ImageDraw
import pytesseract
import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Stephen\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

class ScreenshotTool:
    def __init__(self):
        # Get screen size
        screen_width, screen_height = pyautogui.size()

        self.root = tk.Tk()
        self.root.attributes("-alpha", 0.3)  # Set window transparency (0.0 to 1.0)
        self.root.attributes("-fullscreen", True)  # Set window to fullscreen
        self.root.bind("<ButtonPress-1>", self.on_press)
        self.root.bind("<B1-Motion>", self.on_drag)
        self.root.bind("<ButtonRelease-1>", self.on_release)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.start_x, self.start_y = 0, 0
        self.rect = None
        self.captured_image = None  # Variable to store the captured image

        self.canvas = tk.Canvas(self.root, bg='#E4E4E4', highlightthickness=0, width=screen_width, height=screen_height)
        self.canvas.pack()

    def on_press(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_drag(self, event):
        end_x = event.x_root
        end_y = event.y_root
        self.canvas.coords(self.rect, self.start_x, self.start_y, end_x, end_y)

    def on_release(self, event):
        self.captured_image = self.capture_screenshot()  # Store the captured screenshot
        print("Screenshot captured")
        self.root.destroy()

    def capture_screenshot(self):
        left = min(self.start_x, self.root.winfo_pointerx())
        top = min(self.start_y, self.root.winfo_pointery())
        width = abs(self.root.winfo_pointerx() - self.start_x)
        height = abs(self.root.winfo_pointery() - self.start_y)

        screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
        screenshot.save('screenshot_with_box.png')
        return screenshot

    def on_close(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    screenshot_tool = ScreenshotTool()
    screenshot_tool.run()

    # Access the captured image after the tool has been closed
    if screenshot_tool.captured_image:
        # Do something with the captured image
        img = screenshot_tool.captured_image

    # img = Image.open("screenshot_with_box.png")
    text = pytesseract.image_to_string(img)
    print(len(text))



    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    # embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    # speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # torch.save(speaker_embedding, 'speaker.pt')
    speaker_embedding = torch.load('speaker.pt')
    # You can replace this embedding with your own as well.

    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

    import simpleaudio as sa

    def play_audio(audio_file):
        wave_obj = sa.WaveObject.from_wave_file(audio_file)
        play_obj = wave_obj.play()
        play_obj.wait_done()

    # Replace 'your_audio_file.wav' with the path to your audio file
    audio_file = 'speech.wav'
    play_audio(audio_file)


# import pytesseract
# from PIL import Image
# from transformers import pipeline
# import torch
# import soundfile as sf
# import simpleaudio as sa
# import threading
# from queue import Queue

# def play_audio(audio_file):
#     wave_obj = sa.WaveObject.from_wave_file(audio_file)
#     play_obj = wave_obj.play()
#     play_obj.wait_done()

# def process_and_play_chunk(chunk, synthesiser, speaker_embedding):
#     speech = synthesiser(chunk, forward_params={"speaker_embeddings": speaker_embedding})
#     audio_file = 'speech.wav'
#     sf.write(audio_file, speech["audio"], samplerate=speech["sampling_rate"])
#     play_audio(audio_file)

# if __name__ == "__main__":
#     screenshot_tool = ScreenshotTool()
#     screenshot_tool.run()

#     # Access the captured image after the tool has been closed
#     if screenshot_tool.captured_image:
#         # Do something with the captured image
#         img = screenshot_tool.captured_image

#     # img = Image.open("screenshot_with_box.png")
#     text = pytesseract.image_to_string(img)
#     print(len(text))

#     synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
#     speaker_embedding = torch.load('speaker.pt')

#     chunk_size = 500  # Define your chunk size
#     chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

#     # Create a queue to hold the chunks
#     chunk_queue = Queue()

#     # Put chunks into the queue
#     for chunk in chunks:
#         chunk_queue.put(chunk)

#     # Define a function to process chunks from the queue
#     def process_queue():
#         while not chunk_queue.empty():
#             chunk = chunk_queue.get()
#             process_and_play_chunk(chunk, synthesiser, speaker_embedding)

#     # Create and start the first two threads
#     num_threads = 2  # Number of threads to start initially
#     threads = []
#     for _ in range(num_threads):
#         thread = threading.Thread(target=process_queue)
#         thread.start()
#         threads.append(thread)

#     # Play audio while processing the rest of the chunks
#     while not chunk_queue.empty() or any(thread.is_alive() for thread in threads):
#         if len(threads) < num_threads and not chunk_queue.empty():
#             thread = threading.Thread(target=process_queue)
#             thread.start()
#             threads.append(thread)
#         else:
#             threads[0].join()
#             threads.pop(0)

#     # Wait for all threads to finish
#     for thread in threads:
#         thread.join()

