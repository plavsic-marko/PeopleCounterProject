import cv2
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import config

# Provera da li koristimo GPU i FP16 optimizaciju
device = "cuda" if config.USE_GPU and torch.cuda.is_available() else "cpu"

# Učitavanje YOLOv8m modela sa FP16 optimizacijom
model = YOLO(config.YOLO_MODEL).to(device)  # Koristimo FP32

# Učitavanje videa
cap = cv2.VideoCapture(config.VIDEO_PATH)

# Provera da li je video uspešno učitan
if not cap.isOpened():
    print("❌ Greška pri učitavanju videa!")
    exit()

# Liste za čuvanje podataka
data = []
tracked_people = {}  # ID-evi osoba i njihove poslednje poznate pozicije
appearance_counter = defaultdict(int)  # Broj puta koliko je osoba viđena
next_person_id = 1  # Početni ID za novu osobu
frame_count = 0  

# Otvaramo video za analizu
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("🎬 Kraj videa ili greška pri čitanju frejma.")
        break

    frame_count += 1  

    # YOLO detekcija sa optimizacijama
    results = model(frame, verbose=False, device=device, conf=0.6, imgsz=640, augment=False)

    # Broj ljudi u trenutnom frejmu
    people_count = 0
    new_tracked_people = {}  # Privremena lista za praćenje ljudi u trenutnom frejmu

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = box.conf[0]  
            cls = int(box.cls[0])  

            # Ako je klasa "0" (osoba)
            if cls == 0:
                people_count += 1  

                # Proračun centra bounding box-a
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Proveravamo da li je ovo nova osoba
                new_id = None
                for person_id, (px, py) in tracked_people.items():
                    distance = np.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)

                    # Dinamički prag udaljenosti
                    max_distance = 60 if appearance_counter[person_id] > 15 else 40  
                    
                    if distance < max_distance:
                        new_id = person_id
                        appearance_counter[person_id] += 1  # Povećavamo broj puta viđene osobe
                        break

                # Ako je nova osoba, dodeljujemo joj novi ID
                if new_id is None:
                    new_id = next_person_id
                    next_person_id += 1
                    appearance_counter[new_id] = 1  # Prvi put viđena

                new_tracked_people[new_id] = (center_x, center_y)  # Ažuriramo poziciju osobe

                # Iscrtavanje bounding box-a i ID-a osobe
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {new_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    tracked_people = new_tracked_people  # Ažuriramo praćene ljude

    # Dodajemo podatke u listu
    data.append({"Frame": frame_count, "People Count": people_count})

    # Prikaz broja ljudi u gornjem levom uglu
    cv2.putText(frame, f"People Count: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Prikaz videa sa detekcijom
    cv2.imshow("YOLOv8 People Counter", frame)

    # Usporavanje prikaza videa (ako je prebrz, može se povećati)
    cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()

# Ukupan broj različitih ljudi kroz ceo video
total_people = sum(1 for person_id, count in appearance_counter.items() if count >= 10)  # Osoba mora biti viđena bar 10 frejmova

# Dodavanje ukupnog broja ljudi u CSV fajl
data.append({"Frame": "TOTAL", "People Count": total_people})

# Snimanje podataka u CSV fajl
df = pd.DataFrame(data)
df.to_csv(config.CSV_OUTPUT_PATH, index=False)

print(f"📊 CSV sačuvan u {config.CSV_OUTPUT_PATH}")
print(f"🔢 Ukupan broj ljudi u videu: {total_people}")
