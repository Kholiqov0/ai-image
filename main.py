
 
import cv2
import numpy as np
import requests
import json
import os
import sys
from tkinter import Tk, filedialog
 
 
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME    = "qwen2.5-3b-instruct"
 

 
SYSTEM_PROMPT = """
Ты — помощник, который управляет изображением через OpenCV.
Пользователь пишет команду на русском или английском.
Ты должен вернуть ТОЛЬКО JSON и ничего больше — никаких пояснений, никакого текста вокруг.
 
Поддерживаемые команды:
 
1. Поворот:
   {"action": "rotate", "angle": 90}
   angle — число градусов (положительное = против часовой стрелки)
 
2. Изменение размера:
   {"action": "resize", "scale": 0.5}
   или по пикселям: {"action": "resize", "width": 800, "height": 600}
 
3. Цветовой канал:
   {"action": "channel", "color": "red"}
   color: "red", "green" или "blue"
 
4. Размытие:
   {"action": "blur", "strength": 15}
   strength — нечётное число (7, 11, 15, 21...)
 
5. Отражение:
   {"action": "flip", "direction": "horizontal"}
   direction: "horizontal" или "vertical"
 
6. Чёрно-белый:
   {"action": "grayscale"}
 
7. Яркость:
   {"action": "brightness", "value": 80}
   value от -255 до 255
 
8. Детекция краёв:
   {"action": "edges"}
 
Если команда непонятна:
   {"action": "unknown", "message": "Не понял команду, уточни"}
 
ВАЖНО: только JSON, без markdown, без пояснений.
""".strip()
 

 
def ask_model(user_text: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        "temperature": 0.0,
        "max_tokens": 150,
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("\n❌  Не удалось подключиться к LM Studio.")
        print("    Убедись что:")
        print("    1. LM Studio запущен")
        print("    2. Модель qwen2.5-3b-instruct загружена")
        print("    3. Нажата кнопка 'Start Server' (порт 1234)")
        sys.exit(1)
 
    raw = resp.json()["choices"][0]["message"]["content"].strip()
 
    raw = raw.replace("```json", "").replace("```", "").strip()
 
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {"action": "unknown", "message": f"Модель вернула не JSON: {raw[:80]}"}
    return json.loads(raw[start:end])
 
def process(img: np.ndarray, cmd: dict) -> np.ndarray:
    action = cmd.get("action", "unknown")
 
    if action == "rotate":
        angle = float(cmd.get("angle", 90))
        h, w  = img.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))
 
    elif action == "resize":
        h, w = img.shape[:2]
        if "scale" in cmd:
            s  = float(cmd["scale"])
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
        else:
            nw = int(cmd.get("width",  w))
            nh = int(cmd.get("height", h))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
 
    elif action == "channel":
        color  = cmd.get("color", "red").lower()
        result = np.zeros_like(img)
        ch     = {"blue": 0, "green": 1, "red": 2}.get(color, 2)
        result[:, :, ch] = img[:, :, ch]
        return result
 
    elif action == "blur":
        k = int(cmd.get("strength", 15))
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(img, (k, k), 0)
 
    elif action == "flip":
        direction = cmd.get("direction", "horizontal").lower()
        code = 1 if direction == "horizontal" else 0
        return cv2.flip(img, code)
 
    elif action == "grayscale":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
 
    elif action == "brightness":
        val = int(cmd.get("value", 80))
        return np.clip(img.astype(np.int16) + val, 0, 255).astype(np.uint8)
 
    elif action == "edges":
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
 
    elif action == "unknown":
        raise ValueError(cmd.get("message", "Неизвестная команда"))
 
    else:
        raise ValueError(f"Неизвестный action: {action}")
 
 
def choose_file() -> str:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Все файлы", "*.*")],
    )
    root.destroy()
    return path
 
 
def save_result(img: np.ndarray, source_path: str, action: str) -> str:
    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    base     = os.path.splitext(os.path.basename(source_path))[0]
    out_path = os.path.join(out_dir, f"{base}_{action}.jpg")
    cv2.imwrite(out_path, img)
    return out_path
 
 
def main():
    print("=" * 58)
    print("   CV LM Editor  —  управление фото через Qwen2.5")
    print("=" * 58)
 
    print("\n📂  Откроется окно выбора файла...")
    image_path = choose_file()
    if not image_path:
        print("❌  Файл не выбран.")
        sys.exit(0)
 
    img = cv2.imread(image_path)
    if img is None:
        print("❌  Не удалось открыть изображение.")
        sys.exit(1)
 
    h, w = img.shape[:2]
    print(f"✅  Загружено: {os.path.basename(image_path)}  ({w}×{h})")
    print(f"\n💡  Примеры команд:")
    print("      поверни на 45 градусов")
    print("      уменьши в 2 раза")
    print("      выдели синий канал")
    print("      сделай размытие")
    print("      отрази по вертикали")
    print("      чёрно-белый")
    print("      увеличь яркость на 60")
    print("      выдели края")
    print("\n      (введи 'выход' чтобы завершить)\n")
 
    current_img  = img.copy()
    current_path = image_path
 
    while True:
        user_input = input("🖊  Команда: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("выход", "exit", "quit", "q"):
            print("👋  Завершение.")
            break
 
        print("⏳  Спрашиваю модель...")
        try:
            cmd = ask_model(user_input)
        except json.JSONDecodeError as e:
            print(f"❌  Модель вернула неверный JSON: {e}")
            continue
 
        print(f"🤖  JSON от модели: {json.dumps(cmd, ensure_ascii=False)}")
 
        try:
            result = process(current_img, cmd)
        except ValueError as e:
            print(f"⚠️   {e}")
            continue
 
        out_path = save_result(result, current_path, cmd["action"])
        rh, rw   = result.shape[:2]
        print(f"✅  Готово! {rw}×{rh}  →  {out_path}\n")
 
        current_img  = result
        current_path = out_path
 
 
if __name__ == "__main__":
    main()