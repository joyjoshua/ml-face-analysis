import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import glob
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# "refer images in faces_to_search directory" -> These are the Reference faces (The people you want to find)
REFERENCES_DIR = os.path.join(os.getcwd(), "faces_to_search") 

# "search through target_ref_images" -> These are the Target images (The gallery/camera roll to search in)
SEARCH_IN_DIR = os.path.join(os.getcwd(), "target_ref_imgs")

OUTPUT_DIR = os.path.join(os.getcwd(), "processed_results")
THRESHOLD = 0.5
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_API")

class FaceSearchAgent:
    def __init__(self):
        print("Loading AI Models... (This might take a moment)")
        # Initialize InsightFace
        # Using buffalo_l model which is accurate.
        # providers=['CPUExecutionProvider'] ensures it runs on CPU. 
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces = [] # List of {'name': str, 'embedding': np.array}
        
        if not MAKE_WEBHOOK_URL:
            print("Warning: MAKE_WEBHOOK_URL not found in .env file. Images won't be uploaded.")

    def load_references(self, reference_dir):
        """Generates the 'fingerprint' for the reference persons found in the directory."""
        print(f"Loading reference faces from: {reference_dir}")
        
        if not os.path.exists(reference_dir):
            print(f"Error: Directory '{reference_dir}' not found.")
            return

        # Support various image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        files = []
        for ext in extensions:
            # Recursive=False, we just look in the top folder for now
            files.extend(glob.glob(os.path.join(reference_dir, ext)))
        
        if not files:
            print(f"No images found in {reference_dir}")
            return

        for filepath in files:
            filename = os.path.basename(filepath)
            # Use filename (without extension) as the person's name
            person_name = os.path.splitext(filename)[0]
            
            img = cv2.imread(filepath)
            if img is None:
                print(f"Warning: Could not read image {filename}")
                continue
                
            faces = self.app.get(img)
            if len(faces) == 0:
                print(f"Warning: No face found in reference image {filename}! Skipping.")
                continue
            
            # If multiple faces are found in a reference image, we pick the largest one (presumably the subject)
            # Calculate area (w * h)
            primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            # Normalize the embedding
            embedding = primary_face.embedding / np.linalg.norm(primary_face.embedding)
            
            self.known_faces.append({
                "name": person_name,
                "embedding": embedding
            })
            print(f"[OK] Registered reference: {person_name}")
            
        print(f"Total reference faces loaded: {len(self.known_faces)}")

    def notify_make(self, file_paths):
        """Uploads the processed images to Make.com as a bundle"""
        if not MAKE_WEBHOOK_URL:
            return
        
        if not file_paths:
            print("No matches to upload.")
            return

        print(f"Uploading {len(file_paths)} images to Make.com...", end="\r")
        
        # Prepare valid file paths
        files = []
        opened_files = [] # Keep track of opened files to close them later
        
        try:
            for path in file_paths:
                f = open(path, 'rb')
                opened_files.append(f)
                filename = os.path.basename(path)
                # Key can be 'files' to send as a list, or unique keys. 
                # Sending as 'files' is standard for lists.
                files.append(('files', (filename, f, 'image/jpeg')))
            
            data = {'status': 'completed', 'count': len(file_paths)}
            
            response = requests.post(MAKE_WEBHOOK_URL, files=files, data=data)
            if response.status_code == 200:
                print(f"[OK] Uploaded bundle of {len(file_paths)} images to Make.com")
            else:
                print(f"[Error] Make.com upload failed: {response.text}")
                
        except Exception as e:
            print(f"[Error] Upload connection failed: {e}")
        finally:
            # Close all opened files
            for f in opened_files:
                f.close()

    def process_directory(self, search_dir, output_dir):
        """Iterates through all images in search_dir and finds matches."""
        print(f"\nScanning images in: {search_dir}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(search_dir, ext)))
            
        if not files:
            print("No images found to process.")
            return

        matches_count = 0
        processed_count = 0

        # List to store paths of matched images for batch upload
        matched_images = []

        for filepath in files:
            filename = os.path.basename(filepath)
            print(f"Processing {filename}...", end="\r")
            
            img = cv2.imread(filepath)
            if img is None:
                continue
            
            faces = self.app.get(img)
            
            found_match_in_file = False
            draw_img = img.copy()
            
            for face in faces:
                target_emb = face.embedding / np.linalg.norm(face.embedding)
                
                # Check this face against ALL known references
                best_sim = -1.0
                best_name = "Unknown"
                
                for ref in self.known_faces:
                    sim = np.dot(ref['embedding'], target_emb)
                    if sim > best_sim:
                        best_sim = sim
                        if sim > THRESHOLD:
                            best_name = ref['name']
                
                # Display Logic
                box = face.bbox.astype(int)
                
                if best_sim > THRESHOLD:
                    # MATCH FOUND
                    color = (152, 251, 152) # Light Green
                    label = f"{best_name} ({best_sim:.2f})"
                    found_match_in_file = True

                    # Draw bounding box
                    cv2.rectangle(draw_img, (box[0], box[1]), (box[2], box[3]), color, 2)
                    
                    # Draw label background and text
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    
                    # Ensure label doesn't go off top of image
                    y_label = max(box[1] - 10, 20)
                    
                    cv2.rectangle(draw_img, (box[0], y_label - h - 5), (box[0] + w, y_label + 5), color, -1)
                    cv2.putText(draw_img, label, (box[0], y_label), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            if found_match_in_file:
                # Save the processed image ONLY if a match was found
                output_path = os.path.join(output_dir, f"checked_{filename}")
                cv2.imwrite(output_path, draw_img)
                matches_count += 1
                
                # Add to batch list
                matched_images.append(output_path)
            
            processed_count += 1
        
        # After processing all images, send the bundle
        if matched_images:
            self.notify_make(matched_images)
        
        print(f"\n\n--- Summary ---")
        print(f"Processed: {processed_count} images")
        print(f"Found Matches in: {matches_count} images")
        print(f"Results saved to: {output_dir}")

# --- EXECUTION ---
if __name__ == "__main__":
    agent = FaceSearchAgent()
    
    try:
        # 1. Learn the faces
        agent.load_references(REFERENCES_DIR)
        
        if len(agent.known_faces) == 0:
            print("❌ No reference faces loaded. Please modify 'faces_to_search' directory or add images there.")
        else:
            # 2. Check the gallery
            agent.process_directory(SEARCH_IN_DIR, OUTPUT_DIR)
        
    except Exception as e:
        print(f"An error occurred: {e}")
