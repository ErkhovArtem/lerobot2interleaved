from PIL import Image, ImageDraw, ImageFont
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def visualize_one_episode(repo_id, root, episode_idx):
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    episode_metadata = dataset.meta.episodes[episode_idx]
    
    # Получаем данные из interleaved_instruction
    interleaved_data = episode_metadata['interleaved_instruction']
    image_array = np.array(interleaved_data["image_instruction"][0], dtype=np.uint8)
    
    # Создаем PIL Image
    image = Image.fromarray(image_array)
    
    # Получаем target_object
    target_object = interleaved_data.get('detected_objects', ['Unknown'])[0]
    
    # Создаем копию изображения для рисования
    image_with_text = image.copy()
    draw = ImageDraw.Draw(image_with_text)
    
    try:
        # Пытаемся загрузить шрифт (разные варианты для разных ОС)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
                except:
                    # Fallback на стандартный шрифт
                    font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Получаем размеры текста
    try:
        bbox = draw.textbbox((0, 0), target_object, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        # Если не получилось вычислить размер текста
        text_width = len(target_object) * 10
        text_height = 20
    
    # Позиция текста (верхний левый угол с отступом)
    text_x = 10
    text_y = 10
    
    # Рисуем полупрозрачный фон для текста
    background_coords = [
        text_x - 5, 
        text_y - 5, 
        text_x + text_width + 5, 
        text_y + text_height + 5
    ]
    draw.rectangle(background_coords, fill=(0, 0, 0, 128))  # полупрозрачный черный
    
    # Рисуем текст
    draw.text((text_x, text_y), target_object, fill="white", font=font)
    
    # Добавляем дополнительную информацию
    info_text = f"Episode: {episode_idx}"
    try:
        info_bbox = draw.textbbox((0, 0), info_text, font=font)
        info_width = info_bbox[2] - info_bbox[0]
        info_height = info_bbox[3] - info_bbox[1]
    except:
        info_width = len(info_text) * 8
        info_height = 15
    
    # Фон для дополнительной информации (правый нижний угол)
    info_bg_coords = [
        image.width - info_width - 15,
        image.height - info_height - 10,
        image.width - 5,
        image.height - 2
    ]
    draw.rectangle(info_bg_coords, fill=(0, 0, 0, 128))
    
    # Дополнительный текст
    draw.text(
        (image.width - info_width - 10, image.height - info_height - 5),
        info_text, 
        fill="yellow", 
        font=font
    )
    
    # Сохраняем изображение
    print(f"🎯 Target object: {target_object}")
    print(f"📊 Episode index: {episode_idx}")
    print(f"🖼️  Image size: {image.size}")
    
    # Показываем изображение
    image_with_text.show()

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset episode with interleaved instructions")
    parser.add_argument("--repo-id", required=True, help="Dataset repository ID (e.g., 'lerobot/aloha_static_coffee')")
    parser.add_argument("--root", help="Local root directory for dataset (optional)")
    parser.add_argument("--episode_idx", default=0, type=int, help="Index of episode to show")
    
    args = parser.parse_args()
    
    visualize_one_episode(
        repo_id=args.repo_id,
        root=args.root,
        episode_idx=args.episode_idx,
    )

if __name__ == "__main__":
    main()