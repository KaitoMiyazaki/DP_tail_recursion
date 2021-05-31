import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class TSP: #city(到達地点のx,y座標を持つ2次元のnp.array)
    def __init__(self, city):
        self.city = city
        self.order = list(np.random.permutation(city.shape[0])) #初期フローをランダムに生成
        x = self.city[:,0]
        y = self.city[:,1]
        self.distance_matrix = np.sqrt((x[:,np.newaxis] - x[np.newaxis, :]) ** 2 + 
                                    (y[:,np.newaxis] - y[np.newaxis, :]) ** 2) #cityの各接点間の距離を要素に持つ行列．

    def calculate_total_distace(self, order, distance_matrix):
        """Calculate total distance traveld for given visit order"""
        idx_from = np.array(order) #出発点のリスト
        idx_to = np.array(order[1:] + [order[0]]) #到達点のリスト
        distance_arr = distance_matrix[idx_from, idx_to]

        return np.sum(distance_arr)
    
    def visualize_visit_order(self, order, city_xy):
        """Visualize traveling path for given visit order"""
        route = np.array(order + [order[0]]) #add point of deperture
        x_arr = city_xy[:,0][route]
        y_arr = city_xy[:,1][route]

        plt.figure(figsize=(4,4))
        plt.plot(x_arr, y_arr, 'o-')
        plt.show()

    def solve_with_greedy(self, city):
        N = len(city)
        current_city = 0
        unvisited_cities = set(range(1,N))
        tour = [current_city]

        while unvisited_cities:
            next_city = min(unvisited_cities, key=lambda city: self.distance_matrix[current_city][city])
            unvisited_cities.remove(next_city)
            tour.append(next_city)
            current_city = next_city
        return tour

    def calculate_2opt_exchange_cost(self, visit_order, i, j, distance_matrix):
        """Calculate the difference of cost by applying given 2-opt exchange"""
        n_cities = len(visit_order)
        """a,b,c,d are nodes in the city, b is neighbor of a, and c is to d."""
        a,b = visit_order[i], visit_order[(i + 1) % n_cities]
        c,d = visit_order[j], visit_order[(j + 1) % n_cities]

        """(a,b) + (c,d) と (a,c) + (b,d)を計算，比較．"""
        cost_before = distance_matrix[a,b] + distance_matrix[c,d]
        cost_after = distance_matrix[a,c] + distance_matrix[b,d]

        return cost_after - cost_before
    
    def apply_2opt_exchange(self, visit_order, i, j):
        """Apply 2-opt exchanging on viit order"""
        tmp = visit_order[i + 1 : j + 1] #iからjまでのパス
        tmp.reverse() #iの次に訪れる点がjの時，全体の道順はi-j間を逆に進む．
        visit_order[i + 1 : j + 1] = tmp #tmp に訪問順序iとjを入れ替えた値を保持し，visit_orderのiとjを入れ替える．
        return visit_order

    def improve_with_2opt(self, visit_order, distance_matrix):
        """Check all 2-opt neighbors and improve the visit order"""
        n_cities = len(visit_order)
        cost_diff_best = 0.0
        i_best, j_best = None, None

        for i in range(0, n_cities - 2):
            for j in range(i + 2, n_cities):
                if i == 0 and j == n_cities - 1:
                    continue
                cost_diff = self.calculate_2opt_exchange_cost(visit_order, i, j, distance_matrix)

                if cost_diff < cost_diff_best:
                    cost_diff_best = cost_diff
                    i_best, j_best = i, j
        
        if cost_diff_best < 0.0:
            visit_order_new = self.apply_2opt_exchange(visit_order, i_best, j_best)
            return visit_order_new
        else :
            return None

    def local_search(self, visit_order, distance_matrix, improve_func):
        """Main procedure of local search"""
        #cost_total = self.calculate_total_distace(visit_order, distance_matrix)

        while True:
            improved  = improve_func(visit_order, distance_matrix)
            if not improved: #not None -> True.
                break

            visit_order = improved

        return visit_order

    def solver(self):
        #randomな解
        self.visualize_visit_order(self.order, self.city)
        total_distance = self.calculate_total_distace(self.order, self.distance_matrix)
        print('初期解の総移動距離 = {}'.format(total_distance))

        #greedy-method による解
        tour = self.solve_with_greedy(self.city)
        self.visualize_visit_order(tour, self.city)
        total_distance = self.calculate_total_distace(tour, self.distance_matrix)
        print(f'greedy法適用後の総移動距離 = {total_distance}')

        #初期解:random, 修正法：2-opt
        improved = self.local_search(self.order, self.distance_matrix, self.improve_with_2opt)
        self.visualize_visit_order(improved, self.city)
        total_distance = self.calculate_total_distace(improved, self.distance_matrix)
        print('近傍探索適用後の総移動距離 = {}'.format(total_distance))

        #初期解：greedy-method, 修正法:2-opt
        improved = self.local_search(tour, self.distance_matrix, self.improve_with_2opt)
        self.visualize_visit_order(improved, self.city)
        total_distance = self.calculate_total_distace(improved, self.distance_matrix)
        print('greedy + 2-opt適用後の総移動距離 = {}'.format(total_distance))
def main():
    N = 50
    MAP_SIZE = 100

    np.random.seed(10)
    city_xy = np.random.rand(N, 2) * MAP_SIZE

    my_tsp = TSP(city_xy)
    my_tsp.solver()
if __name__ == "__main__":
    main()