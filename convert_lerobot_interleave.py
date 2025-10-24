"""
Improved script to convert LeRobot dataset to interleave format using official LeRobot tools.
Input: LeRobot dataset (local or from Hugging Face Hub)
Output: LeRobot dataset with interleaved instructions in episodes.jsonl
"""

import os
import json
import pickle
import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tqdm import tqdm
import logging
import shutil

# Import LeRobot dataset class
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_episode, EPISODES_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeRobotInterleaveConverter:
    def __init__(
        self,
        repo_id: str,
        output_dataset_path: str,
        object_list: List[str],
        camera_name: str,
        model_path: str = "google/owlv2-base-patch16-ensemble",
        device: str = "cuda",
        image_size: tuple = (224, 224),
        expand_ratio: float = 0.2,
        confidence_threshold: float = 0.1,
        root: Optional[str] = None
    ):
        self.repo_id = repo_id
        self.output_path = Path(output_dataset_path)
        self.object_list = object_list
        self.camera_name = camera_name
        self.image_size = image_size
        self.expand_ratio = expand_ratio
        self.confidence_threshold = confidence_threshold
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load LeRobot dataset
        logger.info(f"Loading LeRobot dataset: {repo_id}")
        self.dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")

        # Copy dataset to output folder
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        shutil.copytree(self.dataset.root, self.output_path)
        episodes_metadata_path = self.output_path / EPISODES_PATH
        episodes_metadata_path.unlink()
        
        # Initialize models
        logger.info(f"Loading OWLv2 model from {model_path}")
        self.processor = Owlv2Processor.from_pretrained(model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        
        # Image placeholder token
        self.image_placeholder = "<image>"
        
    def expand_box(self, box: List[float], image_width: int, image_height: int) -> List[int]:
        """Expand bounding box by given ratio"""
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        x_expand = width * self.expand_ratio
        y_expand = height * self.expand_ratio
        
        new_x_min = max(0, x_min - x_expand)
        new_y_min = max(0, y_min - y_expand)
        new_x_max = min(image_width, x_max + x_expand)
        new_y_max = min(image_height, y_max + y_expand)
        
        return [int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)]
    
    def find_objects_in_text(self, text: str) -> List[str]:
        """Find objects from object_list that appear in the text"""
        found_objects = []
        text_lower = text.lower()
        
        for obj in self.object_list:
            obj_lower = obj.lower()
            # Simple string matching - can be enhanced with regex for word boundaries
            if obj_lower in text_lower:
                found_objects.append(obj)
        
        return found_objects
    
    def detect_and_crop_objects(self, image: torch.tensor, objects: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Detect objects in image and return cropped images"""
        if not objects:
            return {}
        
        # Клонируем тензор чтобы не изменять оригинал
        img_tensor = image.clone().detach()
        
        # Приводим к диапазону [0, 1] если значения выходят за пределы
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Если значения в диапазоне [0, 255], нормализуем к [0, 1]
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
        
        # Конвертируем в numpy и меняем порядок каналов если нужно
        if img_tensor.shape[0] == 3:  # [C, H, W] -> [H, W, C]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:  # [H, W, C]
            img_np = img_tensor.cpu().numpy()
        
        # Конвертируем в uint8 [0, 255]
        img_np = (img_np * 255).astype(np.uint8)
        
        # Создаем PIL Image
        pil_image = Image.fromarray(img_np)
            
        results = {}
        
        try:
            # Prepare text queries
            text_queries = [[f"a photo of {obj}" for obj in objects]]
            
            # Process image and text
            inputs = self.processor(text=text_queries, images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Run detection
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process results
            target_sizes = torch.Tensor([pil_image.size[::-1]])  # [height, width]
            processed_results = self.processor.post_process_object_detection(
                outputs=outputs, 
                threshold=self.confidence_threshold, 
                target_sizes=target_sizes
            )
            if len(processed_results[0]['scores']) == 0:
                return None
            
            # Find best detection for each object
            best_detections = {}
            for i, (obj, result) in enumerate(zip(objects, processed_results)):
                boxes = result["boxes"].cpu()
                scores = result["scores"].cpu()
                labels = result["labels"].cpu()
                
                for box, score, label in zip(boxes, scores, labels):
                    if score > self.confidence_threshold:
                        box_coords = [round(coord) for coord in box.tolist()]
                        
                        # Check if this is a better detection for this object
                        if obj not in best_detections or score > best_detections[obj]["score"]:
                            best_detections[obj] = {
                                "box": box_coords,
                                "score": score.item()
                            }
            
            # Crop detected objects
            for obj, detection in best_detections.items():
                expanded_box = self.expand_box(
                    detection["box"], 
                    pil_image.width, 
                    pil_image.height
                )
                
                # Crop and resize object
                cropped_obj = pil_image.crop(expanded_box)
                cropped_obj = cropped_obj.resize(self.image_size, Image.Resampling.LANCZOS)
                
                results[obj] = np.array(cropped_obj)
                
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            
        return results
    
    def create_interleaved_instruction(
        self, 
        episode_metadata: Dict[str, Any], 
        detected_objects: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Create interleaved instruction structure"""
        if not detected_objects:
            return None
        
        original_instruction = episode_metadata['tasks'][0]
        # Find object positions in text
        object_positions = []
        current_text = original_instruction
        
        for obj in detected_objects.keys():
            # Find first occurrence of the object in text
            pattern = re.escape(obj)
            match = re.search(pattern, current_text, re.IGNORECASE)
            if match:
                object_positions.append((match.start(), obj))
                # Replace only the first occurrence
                current_text = re.sub(pattern, self.image_placeholder, current_text, count=1, flags=re.IGNORECASE)
        
        # Sort objects by their position in original text
        object_positions.sort(key=lambda x: x[0])
        ordered_objects = [obj for _, obj in object_positions]
        
        # Create image instruction and mask
        image_instruction = []
        image_mask = []
        
        for obj in ordered_objects:
            if obj in detected_objects and detected_objects[obj] is not None:
                image_instruction.append(detected_objects[obj])
                image_mask.append(True)
        
        # Convert numpy arrays to lists for JSON serialization
        image_instruction_serializable = []
        for img_array in image_instruction:
            if img_array is not None:
                # Convert to list and ensure uint8
                img_list = img_array.astype(np.uint8).tolist()
                image_instruction_serializable.append(img_list)
        
        return {"language_instruction": current_text,
        "image_instruction": image_instruction_serializable,
        "image_mask": image_mask,
        "object_order": ordered_objects,
        "detected_objects": list(detected_objects.keys())}
    
    def get_episode_metadata(self, episode_idx: int) -> Dict[str, Any]:
        
        return self.dataset.meta.episodes[episode_idx]
    
    def process_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Process a single episode to add interleaved instructions"""
        episode_metadata = self.get_episode_metadata(episode_idx)        
        
        # Get language instruction
        language_instruction = episode_metadata['tasks'][0]
        
        # Find objects in instruction
        objects_to_find = self.find_objects_in_text(language_instruction)
        if not objects_to_find:
            logger.debug(f"No target objects found in instruction: {language_instruction}")
            return # TODO
        
        logger.info(f"Episode {episode_idx}: Looking for objects {objects_to_find}")
        
        # Detect and crop objects
        detected_objects = None
        frame_idx = self.dataset.episode_data_index['from'][episode_idx].item()
        while detected_objects is None and (frame_idx < self.dataset.episode_data_index['to'][episode_idx].item()):
            image = self.dataset[frame_idx][f'observation.images.{self.camera_name}']
            detected_objects = self.detect_and_crop_objects(image, objects_to_find)
            frame_idx+=5
        
        if not detected_objects:
            logger.info(f"No objects detected in episode {episode_idx}")
            return episode_metadata
        
        # Create interleaved instruction
        interleaved_data = self.create_interleaved_instruction(episode_metadata, detected_objects)
        
        if interleaved_data:
            # Add to episode data
            episode_metadata["interleaved_instruction"] = interleaved_data
            logger.info(f"Added interleaved instruction for episode {episode_idx}")
        
        return episode_metadata
    
    def convert_dataset(self):
        """Convert the entire dataset"""
        logger.info(f"Converting dataset {self.repo_id} to {self.output_path}")
        
        total_episodes = self.dataset.meta.info['total_episodes']
        success_count = 0
        
        for episode_idx in tqdm(range(total_episodes), desc="Processing episodes"):
            try:
                processed_episode_metadata = self.process_episode(episode_idx)
                
                if "interleaved_instruction" in processed_episode_metadata:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing episode {episode_idx}: {e}")
                # Add episode data without interleaved instruction
                processed_episode_metadata = self.get_episode_metadata(episode_idx)
            write_episode(processed_episode_metadata, self.output_path)

        logger.info(f"Conversion complete: {success_count}/{total_episodes} episodes with interleaved instructions")
        logger.info(f"Output dataset saved to {self.output_path}")

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to interleave format")
    parser.add_argument("--repo-id", required=True, help="Dataset repository ID (e.g., 'lerobot/aloha_static_coffee')")
    parser.add_argument("--output", required=True, help="Output dataset path") 
    parser.add_argument("--camera-name", required=True, help="Camera to parse target objects") 
    parser.add_argument("--objects", nargs="+", required=True, help="List of objects to detect")
    parser.add_argument("--root", help="Local root directory for dataset (optional)")
    parser.add_argument("--model", default="google/owlv2-base-patch16-ensemble", help="OWLv2 model path")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    
    args = parser.parse_args()
    
    converter = LeRobotInterleaveConverter(
        repo_id=args.repo_id,
        output_dataset_path=args.output,
        object_list=args.objects,
        root=args.root,
        model_path=args.model,
        device=args.device,
        camera_name=args.camera_name
    )
    
    converter.convert_dataset()

if __name__ == "__main__":
    main()