import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=100):
        print("Algoritmo Genético para Otimização do Corte de Chapa.")
        self.TAM_POP = TAM_POP
        self.initial_layout = recortes_disponiveis  # Available cut parts
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.POP = []
        self.POP_AUX = []
        self.aptidao = []
        self.numero_geracoes = numero_geracoes
        self.initialize_population()
        self.melhor_aptidoes = []
        self.optimized_layout = None  # To be set after optimization

    def initialize_population(self):
        self.POP = []
        for _ in range(self.TAM_POP):
            individual = []
            for recorte in self.initial_layout:
                x = random.randint(0, self.sheet_width - recorte.get("largura", recorte.get("r", 1)*2))
                y = random.randint(0, self.sheet_height - recorte.get("altura", recorte.get("r", 1)*2))
                rotation = random.choice([0, 90]) if "rotacao" in recorte else 0
                individual.append({"tipo": recorte["tipo"], "x": x, "y": y, "rotacao": rotation, **recorte})
            self.POP.append(individual)

    def evaluate(self):
        self.aptidao = []
        for individual in self.POP:
            total_area = sum(self.calculate_area(shape) for shape in individual if self.is_within_bounds(shape))
            self.aptidao.append(total_area)

    def calculate_area(self, shape):
        if shape["tipo"] == "retangular":
            return shape["largura"] * shape["altura"]
        elif shape["tipo"] == "circular":
            return np.pi * (shape["r"] ** 2)
        elif shape["tipo"] == "triangular":
            return (shape["b"] * shape["h"]) / 2
        elif shape["tipo"] == "diamante":
            return (shape["largura"] * shape["altura"]) / 2
        return 0

    def is_within_bounds(self, shape):
        if shape["tipo"] == "retangular":
            return shape["x"] >= 0 and shape["y"] >= 0 and (shape["x"] + shape["largura"] <= self.sheet_width) and (shape["y"] + shape["altura"] <= self.sheet_height)
        elif shape["tipo"] == "circular":
            return shape["x"] - shape["r"] >= 0 and shape["y"] - shape["r"] >= 0 and (shape["x"] + shape["r"] <= self.sheet_width) and (shape["y"] + shape["r"] <= self.sheet_height)
        elif shape["tipo"] == "triangular":
            return shape["x"] >= 0 and shape["y"] >= 0 and (shape["x"] + shape["b"] <= self.sheet_width) and (shape["y"] + shape["h"] <= self.sheet_height)
        elif shape["tipo"] == "diamante":
            return shape["x"] >= 0 and shape["y"] >= 0 and (shape["x"] + shape["largura"] <= self.sheet_width) and (shape["y"] + shape["altura"] <= self.sheet_height)
        return False

    def genetic_operators(self):
        sorted_population = [x for _, x in sorted(zip(self.aptidao, self.POP), key=lambda pair: pair[0], reverse=True)]
        self.POP_AUX = sorted_population[:self.TAM_POP // 2]
        for _ in range(self.TAM_POP - len(self.POP_AUX)):
            p1, p2 = random.sample(self.POP_AUX, 2)
            child = self.crossover(p1, p2)
            self.mutation(child)
            self.POP_AUX.append(child)
        self.POP = self.POP_AUX

    def crossover(self, parent1, parent2):
        crossover_point = len(parent1) // 2
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutation(self, individual):
        for shape in individual:
            if random.random() < 0.1:
                shape["x"] = random.randint(0, self.sheet_width - shape.get("largura", shape.get("r", 1)*2))
                shape["y"] = random.randint(0, self.sheet_height - shape.get("altura", shape.get("r", 1)*2))
                if "rotacao" in shape:
                    shape["rotacao"] = random.choice([0, 90])

    def run(self):
        for _ in range(self.numero_geracoes):
            self.evaluate()
            self.genetic_operators()
        self.optimized_layout = self.POP[0]
        return self.optimized_layout

    def optimize_and_display(self):
        optimized_result = self.run()
        print("Melhor layout encontrado:")
        for shape in optimized_result:
            print(shape)
        return optimized_result
