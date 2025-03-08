import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GeneticAlgorithm:
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=100):
        self.TAM_POP = TAM_POP
        self.initial_layout = recortes_disponiveis
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.numero_geracoes = numero_geracoes
        self.POP = []
        self.aptidao = []
        self.best_solution = None

        self.initialize_population()

    def initialize_population(self):
        """ Inicializa a população evitando sobreposição desde o início """
        self.POP = []
        for _ in range(self.TAM_POP):
            individual = []
            for recorte in self.initial_layout:
                placed = False
                attempts = 0
                while not placed and attempts < 100:
                    x = random.randint(0, self.sheet_width - recorte.get("largura", recorte.get("r", 1)*2))
                    y = random.randint(0, self.sheet_height - recorte.get("altura", recorte.get("r", 1)*2))
                    rotation = random.choice([0, 90]) if "rotacao" in recorte else 0
                    shape = {"tipo": recorte["tipo"], "x": x, "y": y, "rotacao": rotation, **recorte}
                    if self.is_within_bounds(shape) and not self.is_overlapping(shape, individual):
                        individual.append(shape)
                        placed = True
                    attempts += 1
            self.POP.append(individual)

    def calculate_area(self, shape):
        """ Calcula a área de uma forma """
        if shape["tipo"] == "retangular":
            return shape["largura"] * shape["altura"]
        elif shape["tipo"] == "circular":
            return np.pi * (shape["r"] ** 2)
        elif shape["tipo"] == "triangular":
            return (shape["b"] * shape["h"]) / 2
        elif shape["tipo"] == "diamante":
            return shape["largura"] * shape["altura"] / 2
        return 0

    def evaluate(self):
        """ Avaliação otimizada, considerando área útil e sobreposições """
        self.aptidao = []
        for individual in self.POP:
            total_area = sum(self.calculate_area(shape) for shape in individual if self.is_within_bounds(shape))
            overlap_penalty = sum(self.calculate_area(shape) for i, shape in enumerate(individual) if self.is_overlapping(shape, individual[:i]))
            self.aptidao.append(total_area - overlap_penalty)

    def is_within_bounds(self, shape):
        """ Verifica se a forma está dentro da chapa """
        if shape["tipo"] == "retangular":
            return 0 <= shape["x"] <= self.sheet_width - shape["largura"] and 0 <= shape["y"] <= self.sheet_height - shape["altura"]
        elif shape["tipo"] == "circular":
            return shape["x"] - shape["r"] >= 0 and shape["y"] - shape["r"] >= 0 and shape["x"] + shape["r"] <= self.sheet_width and shape["y"] + shape["r"] <= self.sheet_height
        elif shape["tipo"] == "triangular":
            return 0 <= shape["x"] <= self.sheet_width - shape["b"] and 0 <= shape["y"] <= self.sheet_height - shape["h"]
        elif shape["tipo"] == "diamante":
            return 0 <= shape["x"] <= self.sheet_width - shape["largura"] and 0 <= shape["y"] <= self.sheet_height - shape["altura"]
        return False

    def is_overlapping(self, shape, others):
        """ Verifica se a forma se sobrepõe a outras no layout """
        for other in others:
            if shape["tipo"] == "circular" and other["tipo"] == "circular":
                distance = np.hypot(shape["x"] - other["x"], shape["y"] - other["y"])
                if distance < shape["r"] + other["r"]:
                    return True
            else:
                if not (shape["x"] + shape.get("largura", shape.get("r", 1)*2) <= other["x"] or
                        other["x"] + other.get("largura", other.get("r", 1)*2) <= shape["x"] or
                        shape["y"] + shape.get("altura", shape.get("r", 1)*2) <= other["y"] or
                        other["y"] + other.get("altura", other.get("r", 1)*2) <= shape["y"]):
                    return True
        return False

    def genetic_operators(self):
        """ Mantém e melhora a população com crossover e mutação eficientes """
        sorted_population = [x for _, x in sorted(zip(self.aptidao, self.POP), reverse=True)]
        self.POP = sorted_population[:self.TAM_POP // 2]

        while len(self.POP) < self.TAM_POP:
            p1, p2 = random.sample(self.POP, 2)
            child = self.crossover(p1, p2)
            self.mutation(child)
            self.POP.append(child)

    def crossover(self, parent1, parent2):
        """ Realiza crossover sem gerar sobreposições excessivas """
        crossover_point = len(parent1) // 2
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return [shape for shape in child if self.is_within_bounds(shape) and not self.is_overlapping(shape, child)]

    def mutation(self, individual):
        """ Mutação otimizada para não gerar sobreposição """
        for shape in individual:
            if random.random() < 0.1:
                attempts = 0
                while attempts < 50:
                    x = random.randint(0, self.sheet_width - shape.get("largura", shape.get("r", 1)*2))
                    y = random.randint(0, self.sheet_height - shape.get("altura", shape.get("r", 1)*2))
                    if "rotacao" in shape:
                        shape["rotacao"] = random.choice([0, 90])
                    if self.is_within_bounds(shape) and not self.is_overlapping(shape, individual):
                        shape["x"], shape["y"] = x, y
                        break
                    attempts += 1

    def run(self):
        """ Executa o algoritmo e retorna a melhor solução """
        for _ in range(self.numero_geracoes):
            self.evaluate()
            self.genetic_operators()
        self.best_solution = self.POP[0]
        return self.best_solution

    def optimize_and_display(self):
        """ Executa o algoritmo e exibe o melhor layout """
        best_layout = self.run()
        self.plot_layout(best_layout)
        return best_layout

    def plot_layout(self, layout):
        """ Exibe o layout otimizado com todas as formas """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(0, self.sheet_width)
        ax.set_ylim(0, self.sheet_height)

        for shape in layout:
            if shape["tipo"] == "retangular":
                rect = patches.Rectangle((shape["x"], shape["y"]), shape["largura"], shape["altura"], color='blue', alpha=0.5)
                ax.add_patch(rect)
            elif shape["tipo"] == "circular":
                circle = patches.Circle((shape["x"], shape["y"]), shape["r"], color='red', alpha=0.5)
                ax.add_patch(circle)
            elif shape["tipo"] == "diamante":
                diamond = patches.RegularPolygon((shape["x"], shape["y"]), numVertices=4, radius=shape["largura"]/2, color='green', alpha=0.5)
                ax.add_patch(diamond)

        plt.grid(True)
        plt.show()
