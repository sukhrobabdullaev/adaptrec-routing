# -*- coding: utf-8 -*-
"""Komponent profillari va huquqiy toifalar (dissertatsiya, 2.2-bo'lim).

Har bir tavsiya komponenti to'rt o'lchovli profil bilan tavsiflanadi:
    pi_k = (alpha_k, gamma_k, lambda_k, rho_k)
bunda rho_k — ma'lumotning huquqiy (maxfiylik) toifasi. Bu tashkil etuvchi
komponentni texnik emas, yuridik cheklov bo'yicha tavsiflaydi va A7
joylashtirish algoritmi bilan bevosita bog'lanadi.
"""
from dataclasses import dataclass, field
from enum import IntEnum


class HuquqiyToifa(IntEnum):
    """Ma'lumotning huquqiy toifasi (O'zR ZRU-547 asosida, 1.4-bo'lim).

    Qiymat qat'iylik tartibida o'sadi: katta qiymat — qat'iyroq rejim.
    """
    OCHIQ = 0            # ochiq katalog ma'lumoti, cheklovsiz
    SHARTLI = 1          # shaxsga doir, shartli rejimda ishlov beriladi
    LOKAL = 2            # qat'iy lokalizatsiya talab qilinadi


@dataclass
class KomponentProfili:
    """pi_k = (alpha_k, gamma_k, lambda_k, rho_k)."""
    nom: str
    alpha: float                     # kutilayotgan marjinal hissa (dinamik yangilanadi)
    gamma: float                     # hisoblash narxi (shartli birlik)
    lat95: float                     # kechikish, 95-persentil, ms
    rho: HuquqiyToifa = HuquqiyToifa.OCHIQ
    faol: bool = True
    meta: dict = field(default_factory=dict)

    def yangilash_alpha(self, yangi: float, eta: float = 0.05) -> None:
        """Hissani eksponensial silliqlash bilan yangilash (A2 uchun oqim)."""
        self.alpha = (1 - eta) * self.alpha + eta * float(yangi)


def standart_profillar():
    """Dissertatsiyaning asosiy 5 komponentli konfiguratsiyasi (4.2-jadval)."""
    return [
        KomponentProfili("m1 CF/graf",    0.20, 1.0, 15.0, HuquqiyToifa.LOKAL),
        KomponentProfili("m2 kontent",    0.15, 1.5, 20.0, HuquqiyToifa.OCHIQ),
        KomponentProfili("m3 KG",         0.17, 2.0, 25.0, HuquqiyToifa.OCHIQ),
        KomponentProfili("m4 ketma-ket",  0.20, 2.5, 35.0, HuquqiyToifa.LOKAL),
        KomponentProfili("m5 LLM",        0.35, 6.0, 85.0, HuquqiyToifa.SHARTLI),
    ]
