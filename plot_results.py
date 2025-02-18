import pandas as pd
import matplotlib.pyplot as plt
import config

# Učitavanje CSV fajla
df = pd.read_csv(config.CSV_OUTPUT_PATH)

# Provera da li CSV ima podatke
if df.empty:
    print("❌ CSV fajl je prazan! Pokreni main.py da generišeš podatke.")
    exit()

# Crtanje grafikona
plt.figure(figsize=(10, 5))
plt.plot(df["Frame"], df["People Count"], marker="o", linestyle="-", color="b", label="Broj ljudi")

# Postavke grafikona
plt.xlabel("Frejm")
plt.ylabel("Broj ljudi")
plt.title("Promena broja ljudi kroz video")
plt.legend()
plt.grid(True)

# Čuvanje grafikona u fajl
output_graph_path = "output/graph.png"
plt.savefig(output_graph_path)
print(f"📊 Grafikon sačuvan u {output_graph_path}")

# Prikaz grafikona
plt.show()
