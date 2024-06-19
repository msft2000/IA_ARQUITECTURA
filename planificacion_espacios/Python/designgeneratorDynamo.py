# Load the Python Standard and DesignScript Libraries
import sys
import clr
import os

localapp = os.getenv(r'LOCALAPPDATA')

sys.path.append(os.path.join(localapp, r'python-3.9.12-embed-amd64\Lib\site-packages'))

clr.AddReference('ProtoGeometry')
from Autodesk.DesignScript.Geometry import Point as DynPoint, Polygon as DynPolygon, Rectangle as DynRectangle, \
    PolyCurve as DynPolyCurve

from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiPoint, MultiLineString, LinearRing
from shapely import affinity
from scipy.interpolate import interpolate, interp1d
import numpy as np
import copy
import itertools
import random

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
    REPETITION_NUM=1

    # whether the constant score can be used if explicitely indicated
    CAN_USE_CONSTANT_SCORE = True

    def get_item_profitability_ratio(self, item, area_weight):

        """Return the profitability ratio of an item"""

        # the ratio is the value of the item divided by a weighted product of the weight and the shape's area
        return item.value / ((1. - area_weight) * item.weight + area_weight * item.shape.area)

    def get_weighted_sum_of_item_value_and_profitability_ratio(self, item, value_weight, area_weight):

        """Return weighted sum of the value and the profitability ratio of an item"""

        return value_weight * item.value + (1. - value_weight) * self.get_item_profitability_ratio(item, area_weight)

    def select_item(self, items_with_profit_ratio):

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

    def solve_problem(self, problem, greedy_score_function=get_weighted_sum_of_item_value_and_profitability_ratio,
                      value_weight=VALUE_WEIGHT, area_weight=AREA_WEIGHT, max_iter_num=MAX_ITER_NUM,
                      max_iter_num_without_changes=MAX_ITER_NUM_WITHOUT_CHANGES, repetition_num=REPETITION_NUM,
                      item_index_to_place_first=-1, item_specialization_iter_proportion=0.):

        """Find and return a solution to the passed problem, using a greedy strategy"""

        # determine the bounds of the container
        min_x, min_y, max_x, max_y = get_bounds(problem.container.shape)

        max_item_specialization_iter_num = item_specialization_iter_proportion * max_iter_num

        # sort items (with greedy score calculated) by weight, to speed up their discarding (when they would cause the capacity to be exceeded)
        original_items_by_weight = [(index_item_tuple[0],
                                     greedy_score_function(self, index_item_tuple[1], value_weight, area_weight),
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
                    if solution.add_item(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)),
                                         random.choice([0, 90])):

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

            # if the algorithm uses multiple iterations, adopt the current solution as the best one if it is the one with highest value up to now
            if not best_solution or solution.value > best_solution.value:
                best_solution = solution

        # encapsulate all times informatively in a dictionary
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
        return Solution(self.problem,
                        {index: copy.deepcopy(placed_item) for index, placed_item in self.placed_items.items()},
                        self.weight, self.value)

    def is_valid_placement(self, item_index):

        """Se comprueba que la ubicacion de los espacios es valida, considerando al elemento de id  item_index
        y su relacion con el resto de elementos ya colocados en el sitio, sin que se sobremonten entre ellos, exista espacio suficiente
        y no se salga del limite del sitio de planificacion"""

        # No se debe superar el limite de area disponible en el sitio
        if self.weight <= self.problem.container.max_weight:

            shape = self.placed_items[item_index].shape  # poligono del espacio a ubicar

            # Se comprueba que el espacio este dentro de los limites del sitio de planificacion
            if does_shape_contain_other(self.problem.container.shape, shape):

                # Se comprueba si existe interseccion entre los espacios ya colocados en el sitio
                for other_index, other_placed_shape in self.placed_items.items():

                    if item_index != other_index:  # Solo espacios diferentes

                        # Comprobar si existe interseccion entre espacios
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

        # Se recorre la lista de espacios colocados
        for _, placed_shape in self.placed_items.items():

            # Se extraen las coordenadas de ubicacion los extremos del espacio
            min_x, min_y, max_x, max_y = get_bounds(placed_shape.shape)

            # Se buscan los extremos de esas coordenadas
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
            valid_placed_item_indices = [item_index for item_index in self.placed_items.keys() if
                                         item_index not in indices_to_ignore]

        # si no existan elementos para seleccionar
        if not valid_placed_item_indices:
            return None

        # escoger y devolver un indice aleatorio
        return random.choice(valid_placed_item_indices)

    def _add_item(self, item_index, position, rotation):

        """Colocar el espacio en la coordenada y con la rotacion especificada, NO SE VERIFICAN CONFLICTOS NI INTERSECCIONES"""

        # Se marca al espacio como colocado y se guarda su informacion de posicion y rotacion
        self.placed_items[item_index] = PlacedShape(self.problem.items[item_index].id,
                                                    self.problem.items[item_index].shape, position, rotation)

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

    def move_item_in_direction(self, item_index, direction, point_num, min_dist_to_check, max_dist_to_check,
                               has_checked_item_in_container=False):

        """Intentar mover un espacio en una direccion especifica en una cantidad de puntos especificada, validando que no existan intersecciones"""

        # at least one point should be checked
        if point_num >= 1:

            if has_checked_item_in_container or item_index in self.placed_items:

                placed_item = self.placed_items[item_index]

                # normalize the direction
                norm = np.linalg.norm(direction)
                direction = (direction[0] / norm, direction[1] / norm)

                # create a line that goes through the reference position of the item and has the passed direction
                line = LineString([placed_item.position, (placed_item.position[0] + direction[0] * max_dist_to_check,
                                                          placed_item.position[1] + direction[1] * max_dist_to_check)])

                # find the intersection points of the line with other placed items or the container
                intersection_points = list()
                intersection_points.extend(get_intersection_points_between_shapes(line, self.problem.container.shape))
                for other_index, other_placed_shape in self.placed_items.items():
                    if item_index != other_index:
                        intersection_points.extend(
                            get_intersection_points_between_shapes(line, other_placed_shape.shape))

                # at least an intersection should exist
                if intersection_points:

                    # find the smallest euclidean distance from the item's reference position to the first point of intersection
                    intersection_point, min_dist = min(
                        [(p, np.linalg.norm((placed_item.position[0] - p[0], placed_item.position[1] - p[1]))) for p in
                         intersection_points], key=lambda t: t[1])

                    # only proceed if the two points are not too near
                    if min_dist >= min_dist_to_check:
                        points_to_check = list()

                        # if there is only one point to check, just try that one
                        if point_num == 1:
                            return self.move_item_to(item_index, intersection_point)

                        # the segment between the item's reference position and the nearest intersection is divided in a discrete number of points
                        iter_dist = min_dist / point_num
                        for i in range(point_num - 1):
                            points_to_check.append((placed_item.position[0] + direction[0] * i * iter_dist,
                                                    placed_item.position[1] + direction[1] * i * iter_dist))
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

    def _rotate_item_to(self, item_index, new_rotation, has_checked_item_in_container=False,
                        rotate_internal_items=False):

        """Rotsr el espacio hasta alcanzar el nuevo angulo sin verificar conflictos"""

        if has_checked_item_in_container or item_index in self.placed_items:

            old_rotation = self.placed_items[item_index].rotation

            self.placed_items[item_index].rotate_to(new_rotation)

            # if needed, also rotate any items contained in the item of the passed index, with the origin of the shape containing them
            if rotate_internal_items:

                internal_item_indices = self.get_items_inside_item(item_index)

                for internal_index in internal_item_indices:
                    self.placed_items[internal_index].rotate(new_rotation - old_rotation, False,
                                                             self.placed_items[item_index].position)

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

    def show_item(self, item_index, ax, boundary_color, fill_color, container_color, show_item_value_and_weight=False,
                  font=None, show_bounding_box=False, show_reference_position=False, position_offset=(0, 0)):

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
            ax.text(centroid.x + position_offset[0], centroid.y + position_offset[1], value_weight_string,
                    horizontalalignment='center', verticalalignment='center', fontdict=item_font)

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
    # Esta clase representa un espacios ubicarse en el sitio de planificacion

    __slots__ = ("id", "shape", "weight", "value")

    def __init__(self, id, shape, weight, value):
        self.id = id  # Representacion geometrica del espacio
        self.shape = shape  # Representacion geometrica del espacio
        self.weight = weight  # area del epacio
        self.value = value  # peso de adyacencia de ese espacio

    def __deepcopy__(self, memo=None):
        return Item(self.id, copy_shape(self.shape), self.weight, self.value)


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

    __slots__ = ("id", "shape", "position", "rotation")

    def __init__(self, id, shape, position=(0., 0.), rotation=0., move_and_rotate=True):

        """Constructor"""
        self.id = id
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
        return PlacedShape(self.id, self.shape, copy.deepcopy(self.position), self.rotation, False)

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

    def get_id(self):
        return self.id

    def move_to(self, new_position):

        """Move the shape to a new position, updating its points"""

        self.move((new_position[0] - self.position[0], new_position[1] - self.position[1]))

    def rotate(self, angle, update_reference_rotation=True, origin=None):

        """Rotate the shape around its reference position according to the passed rotation angle, expressed in degrees"""

        # only rotate when it makes sense

        if not np.isnan(angle) and angle != 0:
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


def execute_algorithm(algorithm, problem,reps):
    """Execute the passed algorithm as many times as specified (with each execution in a different CPU process if indicated), returning (at least) lists with the obtained solutions, values and elapsed times (one per execution)"""
    solution = algorithm(problem,repetition_num=reps)
    return solution, solution.value


# Poligono que forma el contorno del sitio de planificacion
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


site_process = IN[0]
site = [(tuple([tuple(i) for i in site_process[0]]), [tuple([tuple(k) for k in j]) for j in site_process[1:]])]
# Espacios a ubicar en el sitio de planificaicon

# spaces=[
#     [[ tuplas de coordenadas de la silueta de un obstaculo  ],area de obstaculos,peso_adyacencia],
#       [.....]
# ]

spaces = IN[1]

"""Iniciar semilla"""
random.seed(IN[2])

"""Numero de iteraciones maximas para el algoritmo de ubicacion"""
reps = IN[3]
"""Crear las representaciones geometricas de los objetos"""

# Formar un poligono hueco con los obstaculos del sitio con la libreria shapely

site_polygon = MultiPolygon(site)
max_area = site_polygon.area
site_container = Container(max_area, site_polygon)

# Crear los poligonos de los espacios a ubicar

spaces_items = []  # Guarda los items correspondientes a cada espacio a ubicar
for space in spaces:
    polygon = Polygon(space[0])
    area = space[1]
    ad_score = space[2]
    id = space[3]
    spaces_items.append(Item(id, polygon, area, ad_score))

"""Creacion del disenio"""
problem = Problem(site_container, spaces_items)
solver = RandomDesign(REPETITION_NUM)
algorithm = solver.solve_problem

solution, value = execute_algorithm(algorithm=algorithm, problem=problem,reps)
ids = []
polygons = []
for item in solution.placed_items.values():
    x, y = get_shape_exterior_points(item.shape, True)
    polygons.append(DynPolyCurve.ByPoints([DynPoint.ByCoordinates(i[0], i[1]) for i in list(zip(x, y))]))
    ids.append(item.id)
OUT = {"ids": ids, "polygons": polygons}