from heatmap import HeatMap
from rich.progress import track

heatmap = HeatMap()
# matrix = heatmap.create_from_all_images()
# heatmap.plot(matrix, "fsim_heatmap.png")
for i in track(range(1, 26)):
    if i == 22:
        continue
    matrix = heatmap.create_from_user(i)
    heatmap.plot(matrix, f"heatmaps/fsim_heatmap_user_{i}.png")
