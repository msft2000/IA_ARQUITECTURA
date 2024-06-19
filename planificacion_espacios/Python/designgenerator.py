from shapely.geometry import Polygon, MultiPolygon, Point,LineString,MultiPoint,MultiLineString,LinearRing
from shapely import affinity

import matplotlib.pyplot as plt
from matplotlib import colors, colorbar
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interpolate,interp1d
import numpy as np
from matplotlib.lines import Line2D
import copy
import itertools
import random

#set plotting font and sizes
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cmss10"
plt.rcParams["figure.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

"""Algoritmo de generacion aleatoria"""
class RandomDesign:

    # default weight of the value of an item in the weighted sum of the value and the profitability ratio, that has a weight of 1-VALUE_WEIGHT
    VALUE_WEIGHT = 0.5

    # default weight of the area in the weighted product of item's weight and area; the weight's weight is 1-AREA_WEIGHT
    AREA_WEIGHT = 0.5

    # default maximum number of iterations for the greedy algorithm
    MAX_ITER_NUM = 100000

    # default maximum number of iterations without any change to perform early stopping
    MAX_ITER_NUM_WITHOUT_CHANGES = 30000

    # default number of repetitions of the algorithm
    REPETITION_NUM = 1

    # whether the constant score can be used if explicitely indicated
    CAN_USE_CONSTANT_SCORE = True

    def get_constant_score(self,item, value_weight, area_weight, can_use_constant_score=CAN_USE_CONSTANT_SCORE):

        """Return a constant score regardless of the item's characteristics or any weighting, if allowed"""

        if can_use_constant_score:
            return 1

        return get_weighted_sum_of_item_value_and_profitability_ratio(item, value_weight, area_weight)

    def get_item_profitability_ratio(self, item, area_weight):

        """Return the profitability ratio of an item"""

        # the ratio is the value of the item divided by a weighted product of the weight and the shape's area
        return item.value / ((1. - area_weight) * item.weight + area_weight * item.shape.area)

    def get_weighted_sum_of_item_value_and_profitability_ratio(self,item, value_weight, area_weight):

        """Return weighted sum of the value and the profitability ratio of an item"""

        return value_weight * item.value + (1. - value_weight) * self.get_item_profitability_ratio(item, area_weight)

    def select_item(self,items_with_profit_ratio):

        """Given a list of tuples of the form (item_index, item_profitability_ratio, item), select an item proportionally to its profitability ratio, and return its index"""

        # find the cumulative profitability ratios, code based on random.choices() from the standard library of Python
        cumulative_profit_ratios = list(
            itertools.accumulate(item_with_profit_ratio[1] for item_with_profit_ratio in items_with_profit_ratio))
        profit_ratio_sum = cumulative_profit_ratios[-1]

        # randomly select a ratio within the range of the sum
        profit_ratio = random.uniform(0, profit_ratio_sum)

        # find the value that lies within the random ratio selected; binary search code is based on bisect.bisect_left from standard Python library, but adapted to profitability ratio check
        lowest = 0
        highest = len(items_with_profit_ratio)
        while lowest < highest:
            middle = (lowest + highest) // 2
            if cumulative_profit_ratios[middle] <= profit_ratio:
                lowest = middle + 1
            else:
                highest = middle

        return lowest

    def solve_problem(self,problem, greedy_score_function=get_weighted_sum_of_item_value_and_profitability_ratio,
                      value_weight=VALUE_WEIGHT, area_weight=AREA_WEIGHT, max_iter_num=MAX_ITER_NUM,
                      max_iter_num_without_changes=MAX_ITER_NUM_WITHOUT_CHANGES, repetition_num=REPETITION_NUM,
                      item_index_to_place_first=-1, item_specialization_iter_proportion=0., calculate_times=False,
                      return_value_evolution=False):

        """Find and return a solution to the passed problem, using a greedy strategy"""

        # determine the bounds of the container
        min_x, min_y, max_x, max_y = get_bounds(problem.container.shape)

        max_item_specialization_iter_num = item_specialization_iter_proportion * max_iter_num

        start_time = 0
        sort_time = 0
        item_discarding_time = 0
        item_selection_time = 0
        addition_time = 0
        value_evolution_time = 0

        if calculate_times:
            start_time = time.time()

        if return_value_evolution:
            value_evolution = list()
        else:
            value_evolution = None


        # sort items (with greedy score calculated) by weight, to speed up their discarding (when they would cause the capacity to be exceeded)
        original_items_by_weight = [(index_item_tuple[0],
                                     greedy_score_function(self,index_item_tuple[1], value_weight, area_weight),
                                     index_item_tuple[1]) for index_item_tuple in sorted(list(problem.items.items()),
                                                                                         key=lambda index_item_tuple:
                                                                                         index_item_tuple[1].weight)]


        # discard the items that would make the capacity of the container to be exceeded
        original_items_by_weight = original_items_by_weight[:get_index_after_weight_limit(original_items_by_weight,
                                                                                          problem.container.max_weight)]

        best_solution = None

        # if the algorithm is iterated, it is repeated and the best solution is kept in the end
        for _ in range(repetition_num):

            # if the algorithm is iterated, use a copy of the initial sorted items, to start fresh next time
            if repetition_num > 1:
                items_by_weight = copy.deepcopy(original_items_by_weight)
            else:
                items_by_weight = original_items_by_weight

            # create an initial solution with no item placed in the container
            solution = Solution(problem)

            # placements can only be possible with capacity and valid items
            if problem.container.max_weight and items_by_weight:

                iter_count_without_changes = 0

                # try to add items to the container, for a maximum number of iterations
                for i in range(max_iter_num):


                    # if needed, select a specific item to try to place (only for a maximum number of attempts)
                    if item_index_to_place_first >= 0 and i < max_item_specialization_iter_num:
                        item_index = item_index_to_place_first
                        list_index = -1

                    # perform a random choice of the next item to try to place, weighting each item with their profitability ratio, that acts as an stochastic selection probability
                    else:
                        list_index = self.select_item(items_by_weight)
                        item_index = items_by_weight[list_index][0]


                    # try to add the item in a random position and with a random rotation; if it is valid, remove the item from the pending list
                    if solution.add_item(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)),random.uniform(0, 180)):

                        # the item to place first is assumed to have been placed, if there was any
                        item_index_to_place_first = -1


                        # find the weight that can still be added
                        remaining_weight = problem.container.max_weight - solution.weight

                        # stop early if the capacity has been exactly reached
                        if not remaining_weight:
                            break

                        # remove the placed item from the list of pending items
                        if list_index >= 0:
                            items_by_weight.pop(list_index)

                        # if focusing on an item to place first, find its associated entry in the list to remove it
                        else:
                            for list_i in range(len(items_by_weight)):
                                if items_by_weight[list_i][0] == item_index:
                                    items_by_weight.pop(list_i)
                                    break

                        if calculate_times:
                            start_time = time.time()

                        # discard the items that would make the capacity of the container to be exceeded
                        items_by_weight = items_by_weight[
                                          :get_index_after_weight_limit(items_by_weight, remaining_weight)]


                        # stop early if it is not possible to place more items, because all have been placed or all the items outside would cause the capacity to be exceeded
                        if not items_by_weight:
                            break

                        # reset the potential convergence counter, since an item has been added
                        iter_count_without_changes = 0

                    else:

                        # register the fact of being unable to place an item this iteration
                        iter_count_without_changes += 1

                        # stop early if there have been too many iterations without changes (unless a specific item is tried to be placed first)
                        if iter_count_without_changes >= max_iter_num_without_changes and item_index_to_place_first < 0:
                            break

                    if return_value_evolution:

                        if calculate_times:
                            start_time = time.time()

                        value_evolution.append(solution.value)

            # if the algorithm uses multiple iterations, adopt the current solution as the best one if it is the one with highest value up to now
            if not best_solution or solution.value > best_solution.value:
                best_solution = solution

        # encapsulate all times informatively in a dictionary
        if return_value_evolution:
            return best_solution, value_evolution

        return best_solution


"""Clases a utilizar"""
class Solution(object):

    """Solucion de ubicacion de los espacios dentro del sitio de planificacion"""

    __slots__ = ("problem", "placed_items", "weight", "value")

    def __init__(self, problem, placed_items=None, weight=0., value=0.):


        self.problem = problem
        self.placed_items = placed_items if placed_items else dict()
        self.weight = weight
        self.value = value

    def __deepcopy__(self, memo=None):

        # deep-copy the placed items
        return Solution(self.problem, {index: copy.deepcopy(placed_item) for index, placed_item in self.placed_items.items()}, self.weight, self.value)

    def is_valid_placement(self, item_index):

        """Se comprueba que la ubicacion de los espacios es valida, considerando al elemento de id  item_index
        y su relacion con el resto de elementos ya colocados en el sitio, sin que se sobremonten entre ellos, exista espacio suficiente
        y no se salga del limite del sitio de planificacion"""


        # No se debe superar el limite de area disponible en el sitio
        if self.weight <= self.problem.container.max_weight:

            shape = self.placed_items[item_index].shape #poligono del espacio a ubicar

            #Se comprueba que el espacio este dentro de los limites del sitio de planificacion
            if does_shape_contain_other(self.problem.container.shape, shape):

                # Se comprueba si existe interseccion entre los espacios ya colocados en el sitio
                for other_index, other_placed_shape in self.placed_items.items():

                    if item_index != other_index:#Solo espacios diferentes

                        #Comprobar si existe interseccion entre espacios
                        if do_shapes_intersect(shape, other_placed_shape.shape):

                            return False

                return True

        return False

    def get_area(self):

        """Devuelve la suma de area de los elementos colocados"""

        return sum(placed_shape.shape.area for _, placed_shape in self.placed_items.items())

    def get_global_bounds(self):

        """Devuelve los valores extremos de puntos del sitio de planificacion y los espacios colocados"""

        global_min_x = global_min_y = np.inf
        global_max_x = global_max_y = -np.inf

        #Se recorre la lista de espacios colocados
        for _, placed_shape in self.placed_items.items():

            #Se extraen las coordenadas de ubicacion los extremos del espacio
            min_x, min_y, max_x, max_y = get_bounds(placed_shape.shape)

            #Se buscan los extremos de esas coordenadas
            if min_x < global_min_x:
                global_min_x = min_x

            if min_y < global_min_y:
                global_min_y = min_y

            if max_x > global_max_x:
                global_max_x = max_x

            if max_y > global_max_y:
                global_max_y = max_y

        return global_min_x, global_min_y, global_max_x, global_max_y

    def get_global_bounding_rectangle_area(self):

        """Devuelve el area del espacio formado por las coordenadas maximas y minimas"""

        # Obtiene las coordenadas extremas generales
        min_x, min_y, max_x, max_y = self.get_global_bounds()

        # devuelve el area total del rectangulo formado por las coordenadas extremas
        return abs(min_x - max_x) * abs(min_x - max_y)

    def get_random_placed_item_index(self, indices_to_ignore=None):

        """Devuelve el indice aleatoriamente de un espacio ya ubicado"""

        # Se filtran los espacios para ignorar aquellos que no deben seleccionarse
        if not indices_to_ignore:
            valid_placed_item_indices = list(self.placed_items.keys())
        else:
            valid_placed_item_indices = [item_index for item_index in self.placed_items.keys() if item_index not in indices_to_ignore]

        # si no existan elementos para seleccionar
        if not valid_placed_item_indices:
            return None

        # escoger y devolver un indice aleatorio
        return random.choice(valid_placed_item_indices)

    def _add_item(self, item_index, position, rotation):

        """Colocar el espacio en la coordenada y con la rotacion especificada, NO SE VERIFICAN CONFLICTOS NI INTERSECCIONES"""

        # Se marca al espacio como colocado y se guarda su informacion de posicion y rotacion
        self.placed_items[item_index] = PlacedShape(self.problem.items[item_index].shape, position, rotation)

        # Se actualizan los valores de peso y valor que corresponden al area y score de adyacencia
        self.weight += self.problem.items[item_index].weight
        self.value += self.problem.items[item_index].value

    def add_item(self, item_index, position, rotation=np.nan):

        """Intentar ubicar el espacio en la posicion y rotacion especificada, y comprobar si se logro o no la ubicacion"""

        # SE valida el indice del elemento y este no debe estar dentro del espacio
        if 0 <= item_index < len(self.problem.items) and item_index not in self.placed_items:

            item = self.problem.items[item_index]

            # SE comprueba que el area del espacio no exceda el area disponible del sitio de planificacion
            if self.weight + item.weight <= self.problem.container.max_weight:



                # insertar el espacio en el contenedor sin comprobar inconvenientes
                self._add_item(item_index, position, rotation)

                # Comprobar la validez de la ubicacion
                if self.is_valid_placement(item_index):
                    return True

                # Deshacer la ubicacion hasta encontrar una ubicacion adecuada
                else:
                    self.remove_item(item_index)

        return False

    def remove_item(self, item_index):

        """Intentar quitar el espacio comprobando que este ubicado en el sitio"""

        if item_index in self.placed_items:

            # quitar el area y score de adyacencia colocado
            self.weight -= self.problem.items[item_index].weight
            self.value -= self.problem.items[item_index].value

            # se elimina el elemento
            del self.placed_items[item_index]

            return True

        return False

    def remove_random_item(self):

        """Quitar un espacio aleatoriamente"""

        # if the container is empty, an item index cannot be returned
        if self.weight > 0:

            # escoger un indice aleatorio
            removal_index = self.get_random_placed_item_index()

            # remover el espacio
            if self.remove_item(removal_index):
                return removal_index

        return .1

    def _move_item(self, item_index, displacement, has_checked_item_in_container=False):

        """Mover el espacio del indice una distancia especifica sin verificar la validez de la colocacion"""

        if has_checked_item_in_container or item_index in self.placed_items:

            self.placed_items[item_index].move(displacement)

    def move_item(self, item_index, displacement):

        """Intentar desplazar el espacio una distancia especifica e indicar si se consiguio o no"""

        if item_index in self.placed_items:

            old_position = self.placed_items[item_index].position

            # mover el elemento temporalmente
            self._move_item(item_index, displacement, True)

            # verificar que el movimiento no genere intersecciones
            if self.is_valid_placement(item_index):

                return True

            # se quita el movimiento si se generan intersecciones
            else:

                self._move_item_to(item_index, old_position, True)

        return False

    def _move_item_to(self, item_index, new_position, has_checked_item_in_container=False):

        """Mover el elemento a una posicion especifica, no se verifican intersecciones"""

        if has_checked_item_in_container or item_index in self.placed_items:

            self.placed_items[item_index].move_to(new_position)

    def move_item_to(self, item_index, new_position):

        """Intentar mover un espacio a una posicion especifica, e indicar si no hubieron intersecciones"""

        if item_index in self.placed_items:

            old_position = self.placed_items[item_index].position

            # temporarily move the item, before intersection checks
            self._move_item_to(item_index, new_position)

            # ensure that the solution is valid with the new movement, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the movement if it makes the solution unfeasible
            else:

                self._move_item_to(item_index, old_position)

        return False

    def move_item_in_direction(self, item_index, direction, point_num, min_dist_to_check, max_dist_to_check, has_checked_item_in_container=False):

        """Intentar mover un espacio en una direccion especifica en una cantidad de puntos especificada, validando que no existan intersecciones"""

        # at least one point should be checked
        if point_num >= 1:

            if has_checked_item_in_container or item_index in self.placed_items:

                placed_item = self.placed_items[item_index]

                # normalize the direction
                norm = np.linalg.norm(direction)
                direction = (direction[0] / norm, direction[1] / norm)

                # create a line that goes through the reference position of the item and has the passed direction
                line = LineString([placed_item.position, (placed_item.position[0] + direction[0] * max_dist_to_check, placed_item.position[1] + direction[1] * max_dist_to_check)])

                # find the intersection points of the line with other placed items or the container
                intersection_points = list()
                intersection_points.extend(get_intersection_points_between_shapes(line, self.problem.container.shape))
                for other_index, other_placed_shape in self.placed_items.items():
                    if item_index != other_index:
                        intersection_points.extend(get_intersection_points_between_shapes(line, other_placed_shape.shape))

                # at least an intersection should exist
                if intersection_points:

                    # find the smallest euclidean distance from the item's reference position to the first point of intersection
                    intersection_point, min_dist = min([(p, np.linalg.norm((placed_item.position[0] - p[0], placed_item.position[1] - p[1]))) for p in intersection_points], key=lambda t: t[1])

                    # only proceed if the two points are not too near
                    if min_dist >= min_dist_to_check:
                        points_to_check = list()

                        # if there is only one point to check, just try that one
                        if point_num == 1:
                            return self.move_item_to(item_index, intersection_point)

                        # the segment between the item's reference position and the nearest intersection is divided in a discrete number of points
                        iter_dist = min_dist / point_num
                        for i in range(point_num - 1):
                            points_to_check.append((placed_item.position[0] + direction[0] * i * iter_dist, placed_item.position[1] + direction[1] * i * iter_dist))
                        points_to_check.append(intersection_point)

                        # perform binary search to find the furthest point (among those to check) where the item can be placed in a valid way; binary search code is based on bisect.bisect_left from standard Python library, but adapted to perform placement attempts
                        has_moved = False
                        nearest_point_index = 1
                        furthest_point_index = len(points_to_check)
                        while nearest_point_index < furthest_point_index:
                            middle_point_index = (nearest_point_index + furthest_point_index) // 2
                            if self.move_item_to(item_index, points_to_check[middle_point_index]):
                                nearest_point_index = middle_point_index + 1
                                has_moved = True
                            else:
                                furthest_point_index = middle_point_index

                        return has_moved

        return False

    def _rotate_item(self, item_index, angle, has_checked_item_in_container=False, rotate_internal_items=False):

        """intentar rotar el espacio en el grado especificado, en el punto de rotacion especificado, verificando que no existan conflictos"""

        if has_checked_item_in_container or item_index in self.placed_items:

            self.placed_items[item_index].rotate(angle)

            # if needed, also rotate any items contained in the item of the passed index, with the origin of the shape containing them
            if rotate_internal_items:

                internal_item_indices = self.get_items_inside_item(item_index)

                for internal_index in internal_item_indices:

                    self.placed_items[internal_index].rotate(angle, False, self.placed_items[item_index].position)

    def rotate_item(self, item_index, angle, rotate_internal_items=False):

        """Intentar rotar el espacio en el angulo y en el punto de referencia especificado, e indicar si fue posible"""

        if item_index in self.placed_items:

            old_rotation = self.placed_items[item_index].rotation

            # temporarily rotate the item, before intersection checks
            self._rotate_item(item_index, angle, True, rotate_internal_items)

            # ensure that the solution is valid with the new rotation, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the rotation if it makes the solution unfeasible
            else:

                self._rotate_item_to(item_index, old_rotation, True, rotate_internal_items)

        return False

    def _rotate_item_to(self, item_index, new_rotation, has_checked_item_in_container=False, rotate_internal_items=False):

        """Rotsr el espacio hasta alcanzar el nuevo angulo sin verificar conflictos"""

        if has_checked_item_in_container or item_index in self.placed_items:

            old_rotation = self.placed_items[item_index].rotation

            self.placed_items[item_index].rotate_to(new_rotation)

            # if needed, also rotate any items contained in the item of the passed index, with the origin of the shape containing them
            if rotate_internal_items:

                internal_item_indices = self.get_items_inside_item(item_index)

                for internal_index in internal_item_indices:

                    self.placed_items[internal_index].rotate(new_rotation - old_rotation, False, self.placed_items[item_index].position)

    def rotate_item_to(self, item_index, new_rotation, rotate_internal_items=False):

        """Intetanr rotar el elemento a un nuevo angulo, e indicar si fue posible"""

        if item_index in self.placed_items:

            old_rotation = self.placed_items[item_index].rotation

            # temporarily rotate the item, before intersection checks
            self._rotate_item_to(item_index, new_rotation, rotate_internal_items)

            # ensure that the solution is valid with the new rotation, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the rotation if it makes the solution unfeasible
            else:

                self._rotate_item_to(item_index, old_rotation, rotate_internal_items)

        return False

    def rotate_item_in_direction(self, item_index, clockwise, angle_num):

        """Intentar rotar el espacio en un angulo especificado"""

        has_rotated = False

        if item_index in self.placed_items:

            # calculate the increment in the angle to perform each iteration, to progressively go from an angle greater than 0 to another smaller than 360 (same, and not worth checking since it is the initial state)
            iter_angle = (1 if clockwise else -1) * 360 / (angle_num + 2)
            for _ in range(angle_num):

                # stop as soon as one of the incremental rotations fail; the operation is considered successful if at least one rotation was applied
                if not self.rotate_item(item_index, iter_angle):
                    return has_rotated
                has_rotated = True

        return has_rotated

    def move_and_rotate_item(self, item_index, displacement, angle):

        """Try to move the item with the passed index according to the placed displacement and rotate it as much as indicated by the passed angle"""

        if item_index in self.placed_items:

            old_position = self.placed_items[item_index].position
            old_rotation = self.placed_items[item_index].rotation

            # temporarily move and rotate the item, before intersection checks
            self._move_item(item_index, displacement, True)
            self._rotate_item(item_index, angle, True)

            # ensure that the solution is valid with the new movement and rotation, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the movement and rotation if it makes the solution unfeasible
            else:

                self._move_item_to(item_index, old_position, True)
                self._rotate_item_to(item_index, old_rotation, True)

        return False

    def move_and_rotate_item_to(self, item_index, new_position, new_rotation):

        """Try to move and rotate the item with the passed index so that it has the indicated position and rotation"""

        if item_index in self.placed_items:

            old_position = self.placed_items[item_index].position
            old_rotation = self.placed_items[item_index].rotation

            # temporarily move and rotate the item, before intersection checks
            self._move_item_to(item_index, new_position, True)
            self._rotate_item_to(item_index, new_rotation, True)

            # ensure that the solution is valid with the new movement and rotation, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the movement and rotation if it makes the solution unfeasible
            else:

                self._move_item_to(item_index, old_position, True)
                self._rotate_item_to(item_index, old_rotation, True)

        return False

    def swap_placements(self, item_index0, item_index1, swap_position=True, swap_rotation=True):

        """Try to swap the position and/or the rotation of the two items with the passed indices"""

        # at least position and rotation should be swapped
        if swap_position or swap_rotation:

            # the two items need to be different and placed in the container
            if item_index0 != item_index1 and item_index0 in self.placed_items and item_index1 in self.placed_items:

                # keep track of the original position and rotation of the items
                item0_position = self.placed_items[item_index0].position
                item1_position = self.placed_items[item_index1].position
                item0_rotation = self.placed_items[item_index0].rotation
                item1_rotation = self.placed_items[item_index1].rotation

                # swap position if needed, without checking for validity
                if swap_position:

                    self._move_item_to(item_index0, item1_position, True)
                    self._move_item_to(item_index1, item0_position, True)

                # swap rotation if needed, without checking for validity
                if swap_rotation:

                    self._rotate_item_to(item_index0, item1_rotation, True)
                    self._rotate_item_to(item_index1, item0_rotation, True)

                # ensure that the solution is valid with the swapped movement and/or rotation, i.e. it causes no intersections
                if self.is_valid_placement(item_index0) and self.is_valid_placement(item_index1):

                    return True

                # undo the movement and rotation if it makes the solution unfeasible
                else:

                    # restore position if it was changed
                    if swap_position:

                        self._move_item_to(item_index0, item0_position, True)
                        self._move_item_to(item_index1, item1_position, True)

                    # restore rotation if it was changed
                    if swap_rotation:

                        self._rotate_item_to(item_index0, item0_rotation, True)
                        self._rotate_item_to(item_index1, item1_rotation, True)

        return False

    def get_items_inside_item(self, item_index):

        """Return the indices of the items that are inside the item with the passed index"""

        inside_item_indices = list()

        if item_index in self.placed_items:

            item = self.placed_items[item_index]

            # only multi-polygons can contain other items
            if type(item.shape) == MultiPolygon:

                holes = list()
                for geom in item.shape.geoms:
                    holes.extend(Polygon(hole) for hole in geom.interiors)

                for other_index, placed_shape in self.placed_items.items():

                    if other_index != item_index:

                        for hole in holes:

                            if does_shape_contain_other(hole, self.placed_items[other_index].shape):

                                inside_item_indices.append(other_index)
                                break

        return inside_item_indices

    def visualize(self, title_override=None, show_title=True, show_container_value_and_weight=True, show_outside_value_and_weight=True, show_outside_items=True, color_items_by_profit_ratio=True, show_item_value_and_weight=True, show_value_and_weight_for_container_items=False, show_reference_positions=False, show_bounding_boxes=False, show_value_weight_ratio_bar=True, force_show_color_bar_min_max=False, show_plot=True, save_path=None):

        """Visualize the solution, with placed items in their real position and rotation, and the other ones visible outside the container"""

        can_consider_weight = self.problem.container.max_weight != np.inf

        # set up the plotting figure
        fig_size = (13, 6.75)
        dpi = 160
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        if show_outside_items:
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set(aspect="equal")
            ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
            ax2.set(aspect="equal")
            ax2.tick_params(axis="both", which="major", labelsize=11)
        else:
            ax1 = plt.gca()
            ax1.set(aspect="equal")
            ax2 = None
        ax1.tick_params(axis="both", which="major", labelsize=11)
        if show_title:
            fig.suptitle( "")

        outside_item_bounds = dict()
        total_outside_item_width = 0.

        # represent the container
        x, y = get_shape_exterior_points(self.problem.container.shape, True)
        container_color = (.8, .8, .8)
        boundary_color = (0., 0., 0.)
        ax1.plot(x, y, color=boundary_color, linewidth=1)
        ax1.fill(x, y, color=container_color)
        empty_color = (1., 1., 1.)
        if type(self.problem.container.shape) == MultiPolygon:
            for geom in self.problem.container.shape.geoms:
                for hole in geom.interiors:
                    x, y = get_shape_exterior_points(hole, True)
                    fill_color = empty_color
                    boundary_color = (0., 0., 0.)
                    ax1.plot(x, y, color=boundary_color, linewidth=1)
                    ax1.fill(x, y, color=fill_color)

        font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12}

        # show the total value and weight in the container, and the maximum acceptable weight (capacity)
        if show_container_value_and_weight:
            value_weight_string = "V={}".format(self.value if can_consider_weight else int(self.value))
            if can_consider_weight:
                value_weight_string += ", W={}, Wmax={}".format(self.weight, self.problem.container.max_weight)
            ax1.set_title("Items inside the container\n({})".format(value_weight_string), fontsize=13)

        #determine the range of item profitability ratio, for later coloring of items
        min_profit_ratio = np.inf
        max_profit_ratio = -np.inf
        item_profit_ratios = dict()
        for item_index, item in self.problem.items.items():
            if item.weight == 0:
                profit_ratio = np.inf
            else:
                profit_ratio = item.value / item.weight
            item_profit_ratios[item_index] = profit_ratio
            min_profit_ratio = min(min_profit_ratio, profit_ratio)
            max_profit_ratio = max(max_profit_ratio, profit_ratio)
        best_profit_color = (1, 0.35, 0)
        worst_profit_color = (1, 0.8, 0.8)
        color_interp = interp1d([min_profit_ratio, max_profit_ratio], [0, 1])

        # if possible, add a color-bar showing the value/weight ratio scale

        for item_index, item in self.problem.items.items():
            # represent the placed items
            if item_index in self.placed_items:

                if color_items_by_profit_ratio:
                    fill_color = worst_profit_color + tuple(best_profit_color[i] - worst_profit_color[i] for i in range(len(best_profit_color))) * color_interp(item_profit_ratios[item_index])
                else:
                    fill_color = (1, 0.5, 0.5)

                self.show_item(item_index, ax1, boundary_color, fill_color, container_color, show_item_value_and_weight and show_value_and_weight_for_container_items, font, show_bounding_boxes, show_reference_positions)

            # determine the boundary rectangle of the outside-of-container items
            elif show_outside_items and ax2:

                outside_item_bounds[item_index] = get_bounds(self.problem.items[item_index].shape)
                total_outside_item_width += abs(outside_item_bounds[item_index][2] - outside_item_bounds[item_index][0])

        # show the outside-of-container items
        if show_outside_items and ax2:

            out_value_sum = 0
            out_weight_sum = 0
            row_num = max(1, int(np.log10(len(self.problem.items)) * (3 if len(self.problem.items) < 15 else 4)))
            row = 0
            width = 0
            max_width = 0
            row_height = 0
            height = 0
            for item_index, bounds in outside_item_bounds.items():

                out_value_sum += self.problem.items[item_index].value
                out_weight_sum += self.problem.items[item_index].weight

                if color_items_by_profit_ratio:
                    fill_color = worst_profit_color + tuple(best_profit_color[i] - worst_profit_color[i] for i in range(len(best_profit_color))) * color_interp(item_profit_ratios[item_index])
                else:
                    fill_color = (1, 0.5, 0.5)

                min_x, min_y, max_x, max_y = bounds
                shape_width = abs(max_x - min_x)
                shape_height = abs(max_y - min_y)

                shape_center = get_bounding_rectangle_center(self.problem.items[item_index].shape)
                position_offset = (width + shape_width * 0.5 - shape_center[0], row_height + shape_height * 0.5 - shape_center[1])
                self.show_item(item_index, ax2, boundary_color, fill_color, empty_color, show_item_value_and_weight, font, show_bounding_boxes, show_reference_positions, position_offset)

                height = max(height, row_height + shape_height)

                width += shape_width
                max_width += width
                if width >= total_outside_item_width / row_num:
                    row += 1
                    width = 0
                    row_height = height

            # show the value and weight outside the container
            if show_outside_value_and_weight and ax2:
                value_weight_string = "V={}".format(out_value_sum if can_consider_weight else int(out_value_sum))
                if can_consider_weight:
                    value_weight_string += ", W={}".format(out_weight_sum)
                ax2.set_title("Items outside the container\n({})".format(value_weight_string), fontsize=13)

        fig = plt.gcf()

        if show_plot:
            plt.show()

    def show_item(self, item_index, ax, boundary_color, fill_color, container_color, show_item_value_and_weight=False, font=None, show_bounding_box=False, show_reference_position=False, position_offset=(0, 0)):

        """Show the shape of the passed item index in the indicated axis with the passed colors"""

        if item_index in self.placed_items:
            placed_shape = self.placed_items[item_index]
            shape = placed_shape.shape
        else:
            placed_shape = None
            shape = self.problem.items[item_index].shape

        x, y = get_shape_exterior_points(shape, True)
        if position_offset != (0, 0):
            x = [x_i + position_offset[0] for x_i in x]
            y = [y_i + position_offset[1] for y_i in y]

        ax.plot(x, y, color=boundary_color, linewidth=1)
        ax.fill(x, y, color=fill_color)

        if type(shape) == MultiPolygon:
            for geom in shape.geoms:
                for hole in geom.interiors:
                    x, y = get_shape_exterior_points(hole, True)
                    if position_offset != (0, 0):
                        x = [x_i + position_offset[0] for x_i in x]
                        y = [y_i + position_offset[1] for y_i in y]
                    fill_color = container_color
                    boundary_color = (0., 0., 0.)
                    ax.plot(x, y, color=boundary_color, linewidth=1)
                    ax.fill(x, y, color=fill_color)

        # show the value and weight in the centroid if required
        if show_item_value_and_weight and font:
            centroid = get_centroid(shape)
            value = self.problem.items[item_index].value
            if value / int(value) == 1:
                value = int(value)
            weight = self.problem.items[item_index].weight
            if weight / int(weight) == 1:
                weight = int(weight)
            value_weight_string = "v={}\nw={}".format(value, weight)
            item_font = dict(font)
            item_font['size'] = 9
            ax.text(centroid.x + position_offset[0], centroid.y + position_offset[1], value_weight_string, horizontalalignment='center', verticalalignment='center', fontdict=item_font)

        # show the bounding box and its center if needed
        if show_bounding_box:
            bounds = get_bounds(shape)
            min_x, min_y, max_x, max_y = bounds
            x, y = (min_x, max_x, max_x, min_x, min_x), (min_y, min_y, max_y, max_y, min_y)
            if position_offset != (0, 0):
                x = [x_i + position_offset[0] for x_i in x]
                y = [y_i + position_offset[1] for y_i in y]
            boundary_color = (0.5, 0.5, 0.5)
            ax.plot(x, y, color=boundary_color, linewidth=1)
            bounds_center = get_bounding_rectangle_center(shape)
            ax.plot(bounds_center[0] + position_offset[0], bounds_center[1] + position_offset[1], "r.")

        # show the reference position if required
        if show_reference_position and placed_shape:
            ax.plot(placed_shape.position[0], placed_shape.position[1], "b+")

class Item(object):

    #Esta clase representa un espacios ubicarse en el sitio de planificacion

    __slots__ = ("shape", "weight", "value")

    def __init__(self, shape, weight, value):
        self.shape = shape #Representacion geometrica del espacio
        self.weight = weight # area del epacio
        self.value = value# peso de adyacencia de ese espacio

    def __deepcopy__(self, memo=None):
        return Item(copy_shape(self.shape), self.weight, self.value)

class Container(object):

    """Objeto para representar al sitio de planificacion que contendra los espacios especificados"""

    __slots__ = ("max_weight", "shape")

    def __init__(self, max_weight, shape):

        self.max_weight = max_weight
        self.shape = shape

class Problem(object):

    """Objeto para reprensentar el problema de ubicacion de epacios en combinacion con el problema de la mochila - Knapsack"""

    __slots__ = ("container", "items")

    def __init__(self, container, items):

        self.container = container
        self.items = {index: item for index, item in enumerate(items)}

class PlacedShape(object):

    """Class representing a geometric shape placed in a container with a reference position and rotation"""

    __slots__ = ("shape", "position", "rotation")

    def __init__(self, shape, position=(0., 0.), rotation=0., move_and_rotate=True):

        """Constructor"""

        # the original shape should remain unchanged, while this version is moved and rotated in the 2D space
        self.shape = copy_shape(shape)

        # the original points of the shape only represented distances among them, now the center of the bounding rectangle of the shape should be found in the reference position
        self.position = position
        if move_and_rotate:
            self.update_position(position)
            bounding_rectangle_center = get_bounding_rectangle_center(self.shape)
            self.move((position[0] - bounding_rectangle_center[0], position[1] - bounding_rectangle_center[1]), False)

        # rotate accordingly to the specified angle
        self.rotation = rotation
        if move_and_rotate:
            self.rotate(rotation, False)

    def __deepcopy__(self, memo=None):

        """Return a deep copy"""

        # the constructor already deep-copies the shape
        return PlacedShape(self.shape, copy.deepcopy(self.position), self.rotation, False)

    def update_position(self, new_position):

        """Update the position"""

        self.position = new_position


    def move(self, displacement, update_reference_position=True):

        """Move the shape as much as indicated by the displacement"""

        shape_to_move = self.shape

        # only move when it makes sense

        if displacement != (0., 0.):

            shape_to_move = affinity.translate(shape_to_move, displacement[0], displacement[1])

            self.shape = shape_to_move

            if update_reference_position:
                self.update_position((self.position[0] + displacement[0], self.position[1] + displacement[1]))

    def move_to(self, new_position):

        """Move the shape to a new position, updating its points"""

        self.move((new_position[0] - self.position[0], new_position[1] - self.position[1]))

    def rotate(self, angle, update_reference_rotation=True, origin=None):

        """Rotate the shape around its reference position according to the passed rotation angle, expressed in degrees"""

        # only rotate when it makes sense

        if not np.isnan(angle) and angle != 0 and origin is not None:
            shape_to_rotate = self.shape

            if not origin:
                origin = self.position
            shape_to_rotate = affinity.rotate(shape_to_rotate, angle, origin)

            self.shape = shape_to_rotate

            if update_reference_rotation:
                self.rotation += angle

    def rotate_to(self, new_rotation):

        """Rotate the shape around its reference position so that it ends up having the passed new rotation"""

        self.rotate(new_rotation - self.rotation)

"""Funciones comunes"""

# figure size for plots
PLOT_FIG_SIZE = (16, 9)

# font size for plot titles
PLOT_TITLE_SIZE = 16

# DPI for plots
PLOT_DPI = 200



def get_bounds(shape):

    """Return a tuple with the (min_x, min_y, max_x, max_y) describing the bounding box of the shape"""

    return shape.bounds


def get_bounding_rectangle_center(shape):

    """Return the center of the bounding rectangle for the passed shape"""


    return (shape.bounds[0] + shape.bounds[2]) / 2, (shape.bounds[1] + shape.bounds[3]) / 2


def get_centroid(shape):

    """Return the centroid of a shape"""
    return shape.centroid


def get_shape_exterior_points(shape, is_for_visualization=False):

    """Return the exterior points of a shape"""

    if type(shape) == LinearRing:

        return [coord[0] for coord in shape.coords], [coord[1] for coord in shape.coords]

    if type(shape) == MultiPolygon:

        return shape.geoms[0].exterior.xy

    return shape.exterior.xy


def do_shapes_intersect(shape0, shape1):

    """Return whether the two passed shapes intersect with one another"""
    # default case for native-to-native shape test
    return shape0.intersects(shape1)


def get_intersection_points_between_shapes(shape0, shape1):

    """If the two passed shapes intersect in one or more points (a finite number) return all of them, otherwise return an empty list"""

    intersection_points = list()

    # the contour of polygons and multi-polygons is used for the check, to detect boundary intersection points
    if type(shape0) == Polygon:

        shape0 = shape0.exterior

    elif type(shape0) == MultiPolygon:

        shape0 = MultiLineString(shape0.boundary)

    if type(shape1) == Polygon:

        shape1 = shape1.exterior

    elif type(shape1) == MultiPolygon:

        shape1 = MultiLineString(shape1.boundary)


    intersection_result = shape0.intersection(shape1)

    if intersection_result:

        if type(intersection_result) == Point:
            intersection_points.append((intersection_result.x, intersection_result.y))

        elif type(intersection_result) == MultiPoint:
            for point in intersection_result:
                intersection_points.append((point.x, point.y))

    return intersection_points


def does_shape_contain_other(container_shape, content_shape):

    """Return whether the first shape is a container of the second one, which in such case acts as the content of the first one"""


    return content_shape.within(container_shape)


def copy_shape(shape):

    """Create and return a deep copy of the passed shape"""


    return copy.deepcopy(shape)




def get_index_after_weight_limit(items_by_weight, weight_limit):

    """Given a list of (item_index, [any other fields,] item) tuples sorted by item weight, find the positional index that first exceeds the passed weight limit"""

    # find the item that first exceeds the weight limit; binary search code is based on bisect.bisect_right from standard Python library, but adapted to weight check
    lowest = 0
    highest = len(items_by_weight)
    while lowest < highest:
        middle = (lowest + highest) // 2
        if items_by_weight[middle][-1].weight >= weight_limit:
            highest = middle
        else:
            lowest = middle + 1
    return lowest


def execute_algorithm(algorithm, problem):

    """Execute the passed algorithm as many times as specified (with each execution in a different CPU process if indicated), returning (at least) lists with the obtained solutions, values and elapsed times (one per execution)"""
    solution = algorithm(problem)
    show_solution_plot = True
    if solution and show_solution_plot:
        solution.visualize(show_plot=show_solution_plot, save_path=False)

    return solution, solution.value


#Poligono que forma el contorno del sitio de planificacion
# site=[(
#     (
#         tuplas de coordenadas de la silueta del sitio de planificacion
#     ),
#     [
#         (
#             tuplas de coordenas de la silueta de los obstaculos
#         ),
#           ...
#     ]

# )]


site =[(
        (
            (61.34067803168952, 102.61401318249247), (48.740921904894556, 102.83528689686861), (36.14116577809959, 103.05656061114931), (23.541409651304626, 103.27783432552545), (10.941653524700586, 103.49910803990159), (-1.6581026020943774, 103.72038175427775), (-14.257858728889342, 103.94165546865392), (-26.857614855684304, 104.16292918303006), (-39.457370982288346, 104.3842028974062), (-52.05712710908331, 104.60547661178234), (-64.4961540329102, 103.38488231880544), (-74.06749710017066, 95.52105582203438), (-75.69052553671129, 83.3146157664871), (-75.69052553671129, 70.71291681321759), (-75.69052553671129, 58.111217859852616), (-75.69052553671129, 45.5095189065831), (-75.69052553671129, 32.90781995321812), (-75.69052553671129, 20.306120999948604), (-75.69052553671129, 7.70442204658363), (-75.69052553671129, -4.897276906781343), (-75.69052553671129, -17.498975860050855), (-75.69052553671129, -30.100674813415836), (-75.69052553671129, -42.70237376668535), (-75.69052553690221, -55.30407272005032), (-75.69052553690221, -67.90577167331983), (-75.69052553690221, -80.5074706266848), (-71.81184187833402, -92.27209585584046), (-61.672655435555576, -99.38878935053057), (-49.152136346960404, -100.22086556262846), (-36.55043739378635, -100.2207902000453), (-23.94873844080322, -100.22071483736669), (-11.34703948762916, -100.22063947468807), (1.2546594653539689, -100.22056411200946), (13.85635841852802, -100.22048874933084), (26.458057371702075, -100.22041338665223), (39.05975632468521, -100.2203380239736), (51.66145527785927, -100.22026266129498), (64.24805009347205, -99.94823418515546), (75.19766583767071, -94.15545349133711), (80.51718964772117, -82.96826826585036), (80.66748670032531, -70.37277519372103), (80.66741120190177, -57.77107624064244), (79.85602435408443, -45.310169103001684), (72.83145135505409, -35.05129958145132), (72.48725897630494, -22.4785327586117), (72.4829805275733, -9.876834531606148), (72.47870207903259, 2.7248636953993923), (72.47442363030095, 15.32656192250039), (73.9279157447983, 27.695773031501574), (80.66683933661136, 37.89387982089494), (80.66676397393275, 50.49557877406899), (80.66668861125413, 63.0972777271476), (80.66661324857552, 75.69897668022617), (79.94021701534561, 88.23452490254027), (73.02052288497616, 98.50917293526422)
        ),
        
        
        [
            (
                (-35.70695738417746, 47.769426946501675), (-35.70695738417753, 7.769426946501618), (1.293042615822472, 7.769426946501559), (1.2930426158225394, 47.769426946501554)
            ),
            (
                (62.28526179926068, -8.68622101119283), (62.28526179926065, -23.686221011192828), (72.35115668099482, -23.686221011192863), (72.35115668099485, -8.686221011192861)
            )
        ]
    )]

#Espacios a ubicar en el sitio de planificaicon

# spaces=[
#     [[ tuplas de coordenadas de la silueta de un obstaculo  ],area de obstaculos,peso_adyacencia],
#       [.....]
# ]

spaces=[
    [[(5.0, -15.835197603582008), (5.0, 15.835197603582008), (-5.0, 15.835197603582008), (-5.0, -15.835197603582008)],316.7038, 8.],
[[(11.857265208331842, -6.347347102997883), (11.857265208331842, 6.347347102997883), (-11.857265208331842, 6.347347102997883), (-11.857265208331842, -6.347347102997883)],301.05, 10.],
    [[(15.835197603582008, -5.0), (15.835197603582008, 5.0), (-15.835197603582008, 5.0), (-15.835197603582008, -5.0)],316.7, 8.],
        [[(6.347347102997883, -25.0), (6.347347102997883, 25.0), (-6.347347102997883, 25.0), (-6.347347102997883, -25.0)],634.73, 10.],
    [[(5.0, -15.835197603582008), (5.0, 15.835197603582008), (-5.0, 15.835197603582008), (-5.0, -15.835197603582008)],316.7038, 8.],
[[(11.857265208331842, -6.347347102997883), (11.857265208331842, 6.347347102997883), (-11.857265208331842, 6.347347102997883), (-11.857265208331842, -6.347347102997883)],301.05, 10.],
    [[(15.835197603582008, -5.0), (15.835197603582008, 5.0), (-15.835197603582008, 5.0), (-15.835197603582008, -5.0)],316.7, 8.],
        [[(6.347347102997883, -25.0), (6.347347102997883, 25.0), (-6.347347102997883, 25.0), (-6.347347102997883, -25.0)],634.73, 10.],
    [[(5.0, -15.835197603582008), (5.0, 15.835197603582008), (-5.0, 15.835197603582008), (-5.0, -15.835197603582008)],316.7038, 8.],
[[(11.857265208331842, -6.347347102997883), (11.857265208331842, 6.347347102997883), (-11.857265208331842, 6.347347102997883), (-11.857265208331842, -6.347347102997883)],301.05, 10.],
    [[(15.835197603582008, -5.0), (15.835197603582008, 5.0), (-15.835197603582008, 5.0), (-15.835197603582008, -5.0)],316.7, 8.],
        [[(6.347347102997883, -25.0), (6.347347102997883, 25.0), (-6.347347102997883, 25.0), (-6.347347102997883, -25.0)],634.73, 10.],
    [[(5.0, -15.835197603582008), (5.0, 15.835197603582008), (-5.0, 15.835197603582008), (-5.0, -15.835197603582008)],316.7038, 8.],
[[(11.857265208331842, -6.347347102997883), (11.857265208331842, 6.347347102997883), (-11.857265208331842, 6.347347102997883), (-11.857265208331842, -6.347347102997883)],301.05, 10.],
    [[(15.835197603582008, -5.0), (15.835197603582008, 5.0), (-15.835197603582008, 5.0), (-15.835197603582008, -5.0)],316.7, 8.],
        [[(6.347347102997883, -25.0), (6.347347102997883, 25.0), (-6.347347102997883, 25.0), (-6.347347102997883, -25.0)],634.73, 10.],
    [[(5.0, -15.835197603582008), (5.0, 15.835197603582008), (-5.0, 15.835197603582008), (-5.0, -15.835197603582008)],316.7038, 8.],
[[(11.857265208331842, -6.347347102997883), (11.857265208331842, 6.347347102997883), (-11.857265208331842, 6.347347102997883), (-11.857265208331842, -6.347347102997883)],301.05, 10.],
    [[(15.835197603582008, -5.0), (15.835197603582008, 5.0), (-15.835197603582008, 5.0), (-15.835197603582008, -5.0)],316.7, 8.],
        [[(6.347347102997883, -25.0), (6.347347102997883, 25.0), (-6.347347102997883, 25.0), (-6.347347102997883, -25.0)],634.73, 10.],
    [[(5.0, -15.835197603582008), (5.0, 15.835197603582008), (-5.0, 15.835197603582008), (-5.0, -15.835197603582008)],316.7038, 8.],
[[(11.857265208331842, -6.347347102997883), (11.857265208331842, 6.347347102997883), (-11.857265208331842, 6.347347102997883), (-11.857265208331842, -6.347347102997883)],301.05, 10.],
    [[(15.835197603582008, -5.0), (15.835197603582008, 5.0), (-15.835197603582008, 5.0), (-15.835197603582008, -5.0)],316.7, 8.],
        [[(6.347347102997883, -25.0), (6.347347102997883, 25.0), (-6.347347102997883, 25.0), (-6.347347102997883, -25.0)],634.73, 10.],
    [[(5.0, -15.835197603582008), (5.0, 15.835197603582008), (-5.0, 15.835197603582008), (-5.0, -15.835197603582008)],316.7038, 8.],
[[(11.857265208331842, -6.347347102997883), (11.857265208331842, 6.347347102997883), (-11.857265208331842, 6.347347102997883), (-11.857265208331842, -6.347347102997883)],301.05, 10.],
    [[(15.835197603582008, -5.0), (15.835197603582008, 5.0), (-15.835197603582008, 5.0), (-15.835197603582008, -5.0)],316.7, 8.],
        [[(6.347347102997883, -25.0), (6.347347102997883, 25.0), (-6.347347102997883, 25.0), (-6.347347102997883, -25.0)],634.73, 10.],
    [[(5.0, -15.835197603582008), (5.0, 15.835197603582008), (-5.0, 15.835197603582008), (-5.0, -15.835197603582008)],316.7038, 8.],
[[(11.857265208331842, -6.347347102997883), (11.857265208331842, 6.347347102997883), (-11.857265208331842, 6.347347102997883), (-11.857265208331842, -6.347347102997883)],301.05, 10.],
    [[(15.835197603582008, -5.0), (15.835197603582008, 5.0), (-15.835197603582008, 5.0), (-15.835197603582008, -5.0)],316.7, 8.],
        [[(6.347347102997883, -25.0), (6.347347102997883, 25.0), (-6.347347102997883, 25.0), (-6.347347102997883, -25.0)],634.73, 10.],
    [[(5.0, -15.835197603582008), (5.0, 15.835197603582008), (-5.0, 15.835197603582008), (-5.0, -15.835197603582008)],316.7038, 8.],

]

"""Crear las representaciones geometricas de los objetos"""

#Formar un poligono hueco con los obstaculos del sitio con la libreria shapely

site_polygon= MultiPolygon(site)
max_area=site_polygon.area
site_container=Container(max_area,site_polygon)

#Crear los poligonos de los espacios a ubicar

spaces_items=[]#Guarda los items correspondientes a cada espacio a ubicar
for space in spaces:
    polygon=Polygon(space[0])
    area=space[1]
    ad_score=space[2]
    spaces_items.append(Item(polygon,area,ad_score))

"""Creacion del disenio"""
problem=Problem(site_container,spaces_items)
solution = Solution(problem)
solver=RandomDesign()
algorithm=solver.solve_problem


solution, value=execute_algorithm(algorithm=algorithm, problem=problem)
items=[]
for item in solution.placed_items.values():
    x,y=get_shape_exterior_points(item.shape,True)
    items.append(list(zip(x,y)))

OUTPUT=items