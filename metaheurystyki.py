import time
import random
import numpy as np
import matplotlib.pyplot as plt
import requests
import folium
import pickle


# Dane
cities_coords = {
    # Oryginalne 25
    "Białystok":           (53.1325, 23.1688),
    "Suwałki":             (54.1115, 22.9308),
    "Łomża":               (53.1781, 22.0593),
    "Augustów":            (53.8433, 22.9798),
    "Bielsk Podlaski":     (52.7651, 23.1958),
    "Zambrów":             (52.9856, 22.2450),
    "Grajewo":             (53.6450, 22.4542),
    "Hajnówka":            (52.7416, 23.5173),
    "Sokółka":             (53.4072, 23.5022),
    "Kolno":               (53.4124, 21.9333),
    "Mońki":               (53.4046, 22.7981),
    "Czarna Białostocka":  (53.3056, 23.2801),
    "Wysokie Mazowieckie": (52.9186, 22.5133),
    "Wasilków":            (53.2016, 23.2045),
    "Dąbrowa Białostocka": (53.6521, 23.3516),
    "Sejny":               (54.1065, 23.3496),
    "Choroszcz":           (53.1436, 22.9847),
    "Siemiatycze":         (52.4276, 22.8624),
    "Ciechanowiec":        (52.6826, 22.4975),
    "Supraśl":             (53.2104, 23.3364),
    "Michałowo":           (53.0371, 23.6046),
    "Krynki":              (53.2646, 23.7719),
    "Lipsk":               (53.7336, 23.3986),
    "Suchowola":           (53.5786, 23.1009),
    "Tykocin":             (53.2069, 22.7745),
    "Brańsk":              (52.7441, 22.8339),
    "Szczuczyn":           (53.5628, 22.2858),
    "Knyszyn":             (53.3131, 22.9192),
    "Kleszczele":          (52.5739, 23.3267),
    "Rajgród":             (53.7369, 22.6894),
    "Jedwabne":            (53.2842, 22.3025),
    "Nowogród":            (53.2267, 21.8792),
    "Drohiczyn":           (52.3967, 22.6592),
    "Suraż":               (52.9503, 22.9567),
    "Stawiski":            (53.3794, 22.1558),
    "Goniądz":             (53.4897, 22.7336),
    "Mielnik":             (52.3294, 23.0456),
    "Narew":               (52.9131, 23.5222),
    "Białowieża":          (52.7011, 23.8644),
    "Kuźnica":             (53.5103, 23.6453),
    "Korycin":             (53.4419, 23.0975),
    "Janów":               (53.4697, 23.2217),
    "Szepietowo":          (52.8683, 22.5458),
    "Czyżew":              (52.7961, 22.3275),
    "Boćki":               (52.6517, 23.0369),
    "Śniadowo":            (53.0381, 21.9906),
    "Rutki-Kossaki":       (53.0872, 22.4286),
    "Rutka-Tartak":        (54.3292, 22.9814),
    "Wiżajny":             (54.3828, 22.8697),
    "Giby":                (54.0375, 23.3592)
}

city_names = list(cities_coords.keys())
num_cities = len(city_names)

# Pobieramy dane
coords_str = ";".join([f"{lon},{lat}" for lat, lon in cities_coords.values()])
osrm_url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance"
response = requests.get(osrm_url)
data = response.json()

if data.get("code") == "Ok":
    dist_matrix = np.array(data["distances"], dtype=float)
    dist_matrix = np.nan_to_num(dist_matrix, nan=np.inf)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j and dist_matrix[i, j] == 0.0:
                dist_matrix[i, j] = np.inf
    print(f"--Macierz dla {num_cities} miast gotowa!--")
else:
    raise ValueError(f"Błąd API: {data.get('message', 'Nieznany błąd')}")

# Zdefiniowanie wspólnych funkci i stałych
TIME_LIMIT = 5

def calculate_distance(tour, dm):
    dist = np.sum(dm[tour[:-1], tour[1:]])
    dist += dm[tour[-1], tour[0]]
    return dist

def two_opt_swap(tour, i, j):
    new_tour = tour.copy()
    new_tour[i:j+1] = new_tour[i:j+1][::-1]
    return new_tour

def get_random_tour(n, start_node):
    cities = list(range(n))
    cities.remove(start_node)
    random.shuffle(cities)
    return [start_node] + cities

# zachłanny najbliższy sąsiad
def solve_nn(dm, start_node):
    start_t = time.time()
    n = dm.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start_node)
    tour = [start_node]
    curr_node = start_node
    while unvisited:
        next_node = min(unvisited, key=lambda v: dm[curr_node, v])
        tour.append(next_node)
        unvisited.remove(next_node)
        curr_node = next_node
    final_dist = calculate_distance(tour, dm)
    return tour, final_dist, [(time.time() - start_t, final_dist)]

# Lista tabu
def solve_tabu(dm, start_node, time_limit):
    start_t = time.time()
    n = dm.shape[0]
    current_tour = get_random_tour(n, start_node)
    best_tour = current_tour.copy()
    best_dist = calculate_distance(best_tour, dm)
    
    tabu_list = []
    tabu_tenure = 25 
    
    history = [(time.time() - start_t, best_dist)]
    iteration = 0
    iters_no_improve = 0

    while time.time() - start_t < time_limit:
        iteration += 1
        iters_no_improve += 1
        best_neighbor, best_neighbor_dist, best_swap = [], float('inf'), (0, 0)
        
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                candidate = two_opt_swap(current_tour, i, j)
                cand_dist = calculate_distance(candidate, dm)
                swap_move = (i, j)
                
                is_tabu = swap_move in tabu_list
                
                if (not is_tabu or cand_dist < best_dist) and cand_dist < best_neighbor_dist:
                    best_neighbor, best_neighbor_dist, best_swap = candidate, cand_dist, swap_move
                    
        if not best_neighbor: break
        current_tour = best_neighbor
        
        # FIFO
        tabu_list.append(best_swap)
        
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0) 
            
        if best_neighbor_dist < best_dist:
            best_dist, best_tour = best_neighbor_dist, current_tour.copy()
            history.append((time.time() - start_t, best_dist))
            iters_no_improve = 0
            
        if iters_no_improve > 500: 
            current_tour = get_random_tour(n, start_node)
            iters_no_improve = 0
            
    history.append((time.time() - start_t, best_dist))
    return best_tour, best_dist, history

# Algorytm genetyczny
def tournament_select(pop, fit):
    best_idx = max(random.sample(range(len(pop)), 3), key=lambda i: fit[i])
    return pop[best_idx]

def solve_ga(dm, start_node, time_limit):
    start_t = time.time()
    n = dm.shape[0]
    pop_size = 150 
    population = [get_random_tour(n, start_node) for _ in range(pop_size)]
    best_tour, best_dist = population[0].copy(), float('inf')
    history = []
    gens_no_improve = 0

    while time.time() - start_t < time_limit:
        gens_no_improve += 1
        fitness = [1.0 / calculate_distance(ind, dm) for ind in population]
        new_population = [None] * pop_size
        best_idx = int(np.argmax(fitness))
        new_population[0] = population[best_idx].copy()
        if calculate_distance(population[best_idx], dm) < best_dist:
            best_dist, best_tour = calculate_distance(population[best_idx], dm), population[best_idx].copy()
            history.append((time.time() - start_t, best_dist))
            gens_no_improve = 0
        if gens_no_improve > 150:
            for i in range(1, pop_size // 2): population[i] = get_random_tour(n, start_node)
            gens_no_improve = 0
        for p in range(1, pop_size):
            p1, p2 = tournament_select(population, fitness), tournament_select(population, fitness)
            s_cut, e_cut = sorted(random.sample(range(1, n), 2))
            child = [-1] * n
            child[0] = start_node
            child[s_cut:e_cut+1] = p1[s_cut:e_cut+1]
            idx = 1
            for val in p2[1:]:
                if val not in child:
                    while idx < n and child[idx] != -1: idx += 1
                    if idx < n: child[idx] = val
            if -1 in child: child = get_random_tour(n, start_node)
            if random.random() < 0.4: 
                m1, m2 = random.sample(range(1, n), 2)
                child[m1], child[m2] = child[m2], child[m1]
            new_population[p] = child
        population = new_population
        
    history.append((time.time() - start_t, best_dist))
    return best_tour, best_dist, history

# Wyżarzanie symulowane
def solve_sa(dm, start_node, time_limit, initial_tour=None):
    start_t = time.time()
    n = dm.shape[0]
    current_tour = get_random_tour(n, start_node) if initial_tour is None else initial_tour.copy()
    current_dist = calculate_distance(current_tour, dm)
    best_tour, best_dist = current_tour.copy(), current_dist
    T, cooling_rate = 50000.0, (1.0 / 50000.0) ** (1.0 / (time_limit * 5_000)) 
    history = [(time.time() - start_t, best_dist)]
    
    while time.time() - start_t < time_limit:
        i, j = sorted(random.sample(range(1, n), 2))
        candidate = two_opt_swap(current_tour, i, j)
        cand_dist = calculate_distance(candidate, dm)
        delta = cand_dist - current_dist
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_tour, current_dist = candidate, cand_dist
            if current_dist < best_dist:
                best_dist, best_tour = current_dist, current_tour.copy()
                history.append((time.time() - start_t, best_dist))
        T *= cooling_rate
        
    history.append((time.time() - start_t, best_dist))
    return best_tour, best_dist, history


# Optymalizacja
nn_tour, nn_dist, nn_hist = solve_nn(dist_matrix, 0)
print(f"1. Nearest Neighbor: {nn_dist/1000:.2f} km")

tabu_tour, tabu_dist, tabu_hist = solve_tabu(dist_matrix, 0, TIME_LIMIT)
print(f"2. Tabu Search     : {tabu_dist/1000:.2f} km")

ga_tour, ga_dist, ga_hist = solve_ga(dist_matrix, 0, TIME_LIMIT)
print(f"3. Genetyczny      : {ga_dist/1000:.2f} km")

sa_tour, sa_dist, sa_hist = solve_sa(dist_matrix, 0, TIME_LIMIT)
print(f"4. Wyżarzanie      : {sa_dist/1000:.2f} km")

# Wykresy zbieżności do optimum

def unpack_history(hist):
    return [h[0] for h in hist], [h[1] / 1000 for h in hist] # od razu w km

t_nn, d_nn = unpack_history(nn_hist)
t_tabu, d_tabu = unpack_history(tabu_hist)
t_ga, d_ga = unpack_history(ga_hist)
t_sa, d_sa = unpack_history(sa_hist)

fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
fig.suptitle(f"Indywidualna zbieżność algorytmów TSP ({num_cities} miast)", fontsize=16)

# Sąsiad
axs[0, 0].axhline(y=d_nn[0], color='blue', linestyle='--', linewidth=2)
axs[0, 0].set_title(f"1. Nearest Neighbor ({d_nn[0]:.1f} km)")
axs[0, 0].set_xlabel("Czas [s]")
axs[0, 0].set_ylabel("Dystans [km]")
axs[0, 0].grid(True, linestyle='--', alpha=0.7)

# Tabu
axs[0, 1].step(t_tabu, d_tabu, where='post', color='red', linewidth=2)
axs[0, 1].set_title(f"2. Tabu Search ({d_tabu[-1]:.1f} km)")
axs[0, 1].set_xlabel("Czas [s]")
axs[0, 1].set_ylabel("Dystans [km]")
axs[0, 1].grid(True, linestyle='--', alpha=0.7)

# Genetyka
axs[1, 0].step(t_ga, d_ga, where='post', color='green', linewidth=2)
axs[1, 0].set_title(f"3. Algorytm Genetyczny ({d_ga[-1]:.1f} km)")
axs[1, 0].set_xlabel("Czas [s]")
axs[1, 0].set_ylabel("Dystans [km]")
axs[1, 0].grid(True, linestyle='--', alpha=0.7)

# Wyżarzanie
axs[1, 1].step(t_sa, d_sa, where='post', color='orange', linewidth=2)
axs[1, 1].set_title(f"4. Wyżarzanie Symulowane ({d_sa[-1]:.1f} km)")
axs[1, 1].set_xlabel("Czas [s]")
axs[1, 1].set_ylabel("Dystans [km]")
axs[1, 1].grid(True, linestyle='--', alpha=0.7)

# Wykresy i wizualizacja ścieżek 

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
chart_individual_file = "Wykresy_Indywidualne_50_Miast.png"
plt.savefig(chart_individual_file)
print(f"Zapisano wykresy indywidualne jako '{chart_individual_file}'.")
plt.show() 

plt.figure(figsize=(12, 7), dpi=150)
plt.title(f"Zbiorcza Zbieżność Algorytmów TSP ({num_cities} miast)")
plt.xlabel("Czas obliczeń [s]")
plt.ylabel("Całkowity Dystans [km]")

plt.axhline(y=d_nn[0], color='blue', linestyle='--', label=f"Nearest Neighbor ({d_nn[0]:.1f} km)", linewidth=2)

plt.step(t_tabu, d_tabu, where='post', color='red', label=f"Tabu Search ({d_tabu[-1]:.1f} km)", linewidth=2)
plt.step(t_ga, d_ga, where='post', color='green', label=f"Genetyczny ({d_ga[-1]:.1f} km)", linewidth=2)
plt.step(t_sa, d_sa, where='post', color='orange', label=f"Wyżarzanie ({d_sa[-1]:.1f} km)", linewidth=2)

plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

chart_combined_file = "Wykres_Zbiorczy_50_Miast.png"
plt.savefig(chart_combined_file)
print(f"Zapisano wykres zbiorczy jako '{chart_combined_file}'. Zamykanie okna uruchomi tworzenie mapy HTML...")
plt.show()

print("\nGenerowanie interaktywnej mapy z trasami...")

start_lat, start_lon = cities_coords["Białystok"]
m = folium.Map(location=[start_lat, start_lon], zoom_start=8, tiles='CartoDB positron')

cities_group = folium.FeatureGroup(name=f"Miasta ({num_cities})", show=True)
for name, coords in cities_coords.items():
    if name == "Białystok":
        folium.Marker(location=coords, popup="<b>Białystok</b> (START)", 
                      icon=folium.Icon(color="black", icon="star")).add_to(cities_group)
    else:
        folium.CircleMarker(location=coords, radius=4, color="black", fill=True, 
                            fill_opacity=0.8, popup=name).add_to(cities_group)
cities_group.add_to(m)

algorithms_data = [
    ("1. Nearest Neighbor (Baseline)", nn_tour, "blue", False, 6),
    ("2. Tabu Search", tabu_tour, "red", False, 5),
    ("3. Algorytm Genetyczny", ga_tour, "green", False, 4),
    ("4. Wyżarzanie Symulowane (Najlepsza)", sa_tour, "orange", True, 3) 
]

for name, tour, color, show_by_default, weight in algorithms_data:
    fg = folium.FeatureGroup(name=name, show=show_by_default)
    
    route_coords = [cities_coords[city_names[i]] for i in tour]
    route_coords.append(route_coords[0]) 

    ordered_str = ";".join([f"{lon},{lat}" for lat, lon in route_coords])
    route_url = f"http://router.project-osrm.org/route/v1/driving/{ordered_str}?geometries=geojson"
    
    try:
        time.sleep(1.0)
        route_response = requests.get(route_url).json()
        geometry_coords = route_response['routes'][0]['geometry']['coordinates']
        road_path = [[lat, lon] for lon, lat in geometry_coords]
        folium.PolyLine(road_path, color=color, weight=weight, opacity=0.9).add_to(fg)
    except Exception as e:
        print(f"Uwaga: Nie udało się pobrać geometrii dla {name}, rysuję linie proste.")
        folium.PolyLine(route_coords, color=color, weight=weight, opacity=0.9, dash_array='5, 5').add_to(fg)
    
    fg.add_to(m)

folium.LayerControl(position='topright', collapsed=False).add_to(m)

map_file = "Porownanie_Tras_Podlasie_50_Miast.html"
m.save(map_file)
print(f"\nGOTOWE! Otwórz plik '{map_file}' w przeglądarce.")





print("\nZapisywanie wyników do pliku pickle...")

wyniki_tsp = {
    "lista_miast": city_names,
    "wspolrzedne": cities_coords,
    "NN": {
        "trasa": nn_tour,
        "dystans": nn_dist,
        "historia": nn_hist
    },
    "Tabu": {
        "trasa": tabu_tour,
        "dystans": tabu_dist,
        "historia": tabu_hist
    },
    "GA": {
        "trasa": ga_tour,
        "dystans": ga_dist,
        "historia": ga_hist
    },
    "SA": {
        "trasa": sa_tour,
        "dystans": sa_dist,
        "historia": sa_hist
    }
}

plik_pickle = "wyniki_tsp_50_miast.pkl"

with open(plik_pickle, "wb") as f:
    pickle.dump(wyniki_tsp, f)

print(f"Pomyślnie zapisano wszystkie dane do pliku: '{plik_pickle}'.")




