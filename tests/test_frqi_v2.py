"""Regressoes dos erros de escala, endereco e leitura de medidas do FRQI."""

import unittest

import numpy as np
from qiskit.quantum_info import Statevector

from frqi_v2 import (
    circuito_frqi, executar_experimento, imagem_demo, metricas,
    reconstruir_ideal, reconstruir_imagem,
)


class TestFRQI(unittest.TestCase):
    def test_amplitudes_da_grade_assimetrica_correspondem_a_equacao_frqi(self):
        imagem = imagem_demo(8)
        estado = Statevector.from_instruction(circuito_frqi(imagem)).data
        esperado = np.empty(2 * imagem.size, dtype=complex)
        angulos = imagem.ravel() * np.pi / 2
        esperado[0::2] = np.cos(angulos) / np.sqrt(imagem.size)
        esperado[1::2] = np.sin(angulos) / np.sqrt(imagem.size)
        np.testing.assert_allclose(estado, esperado, atol=1e-12, rtol=1e-12)

    def test_reconstrucao_ideal_preserva_cada_intensidade_e_posicao(self):
        imagem = imagem_demo(8)
        ideal = reconstruir_ideal(circuito_frqi(imagem), imagem.shape)
        np.testing.assert_allclose(ideal, imagem, atol=1e-10, rtol=0)

    def test_um_pixel_sem_qubits_de_posicao(self):
        for valor in (0, 0.25, 0.5, 0.75, 1):
            with self.subTest(valor=valor):
                imagem = np.array([[valor]])
                ideal = reconstruir_ideal(circuito_frqi(imagem), imagem.shape)
                np.testing.assert_allclose(ideal, imagem, atol=1e-12)

    def test_aer_preserva_posicoes_de_preto_e_branco(self):
        # Marcadores que nao sobrevivem a uma inversao dos seis bits de endereco.
        imagem = np.zeros((8, 8))
        imagem[0, 1] = 1
        imagem[2, 3] = 1
        imagem[7, 5] = 1
        resultado = executar_experimento(imagem, shots=6400, seed=42)
        self.assertTrue(np.all(resultado["amostras"] > 0))
        self.assertEqual(int(resultado["amostras"].sum()), 6400)
        np.testing.assert_allclose(resultado["medida"], imagem, atol=1e-12)

    def test_aer_estima_tons_de_cinza_sem_erro_sistematico(self):
        imagem = imagem_demo(8)
        resultado = executar_experimento(imagem, shots=64000, seed=42)
        self.assertLess(resultado["metricas_medida"]["mse"], 0.001)
        self.assertLess(resultado["metricas_medida"]["erro_maximo"], 0.06)
        self.assertEqual(resultado["metricas_medida"]["cobertura"], 1)

    def test_pixel_nao_observado_e_desconhecido_em_vez_de_preto(self):
        reconstruida, amostras = reconstruir_imagem({"001": 100}, (2, 2))
        self.assertEqual(reconstruida[0, 0], 1)
        self.assertTrue(np.all(np.isnan(reconstruida.ravel()[1:])))
        self.assertEqual(int(amostras.sum()), 100)
        m = metricas(np.ones((2, 2)), reconstruida)
        self.assertEqual(m["cobertura"], 0.25)
        self.assertIsNone(m["mse"])
        self.assertIsNone(m["psnr_db"])

    def test_leitura_de_cor_e_posicao_inclui_preto_e_cinza(self):
        # 00: preto; 01: branco; 10: p=1/2 (cinza 0.5); 11: p=1/4 (cinza 1/3).
        counts = {"000": 100, "011": 100, "100": 50, "101": 50, "110": 75, "111": 25}
        imagem, amostras = reconstruir_imagem(counts, (2, 2))
        np.testing.assert_allclose(imagem, [[0, 1], [0.5, 1 / 3]], atol=1e-12)
        np.testing.assert_array_equal(amostras, np.full((2, 2), 100))

    def test_rejeita_entradas_invalidas(self):
        for imagem in (np.zeros((3, 3)), np.array([[np.nan]]),
                       np.array([[1.1]]), np.array([[-0.1]]), np.zeros(8)):
            with self.subTest(imagem=imagem):
                with self.assertRaises(ValueError):
                    circuito_frqi(imagem)
        for counts in ({"000": -1}, {"000": 1.5}, {"11x": 1}, {"0000": 1}):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    reconstruir_imagem(counts, (2, 2))
        with self.assertRaises(ValueError):
            executar_experimento(np.zeros((2, 2)), shots=0)


if __name__ == "__main__":
    unittest.main()
