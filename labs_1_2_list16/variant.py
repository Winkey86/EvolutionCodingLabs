"""Расчёт индивидуального варианта по методическим указаниям."""

from __future__ import annotations

from dataclasses import dataclass


GROUP_INDEX = {507: 0, 517: 1, 527: 2}


@dataclass(frozen=True)
class Variant:
    list_number: int
    group: int
    year_suffix: int
    number: int
    dimension: int


def calculate_variant(list_number: int, group: int, year_suffix: int = 26) -> Variant:
    if group not in GROUP_INDEX:
        allowed = ", ".join(map(str, GROUP_INDEX))
        raise ValueError(f"Неизвестная группа {group}. Допустимые значения: {allowed}")
    if list_number < 1:
        raise ValueError("Номер в списке должен быть положительным")

    q = GROUP_INDEX[group]
    number = 1 + ((17 * list_number + 7 * q + year_suffix) % 20)
    dimension = 5 + number % 6
    return Variant(list_number, group, year_suffix, number, dimension)


def possible_variants(list_number: int = 16, year_suffix: int = 26) -> list[Variant]:
    return [calculate_variant(list_number, group, year_suffix) for group in GROUP_INDEX]


if __name__ == "__main__":
    for item in possible_variants():
        print(
            f"Группа {item.group}: V={item.number}, "
            f"размерность лабораторной №1 d={item.dimension}"
        )
