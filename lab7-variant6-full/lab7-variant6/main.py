"""
Лабораторная работа № 7. Вариант 6.

Рекурсивные алгоритмы и динамическое программирование.

Параметры варианта 6 (табл. 1):
  - Задача А: биномиальный коэффициент C(n, k), n до 28
    три реализации: наивная рекурсия, мемоизация, итеративная
  - Задача Б: максимальная сумма подотрезка, массив из 10 000 чисел
  - Доп. Г6: сравнить свою мемоизацию с functools.lru_cache

Дисциплина: Основы алгоритмизации и программирование
Студент: Голдаева Валерия Валентиновна, группа БИ 2-1
"""

# ---------------------------------------------------------------------------
# Стандартные библиотеки
# ---------------------------------------------------------------------------
import functools      # lru_cache для доп. задания Г6
import random         # генерация тестового массива
import time           # time.perf_counter
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Глобальный счётчик вызовов (сбрасывается перед каждым замером)
# ---------------------------------------------------------------------------
calls: int = 0


# ===========================================================================
# 1. ЗАДАЧА А: БИНОМИАЛЬНЫЙ КОЭФФИЦИЕНТ C(n, k)
# ===========================================================================
#
# Рекуррентное соотношение:
#   C(n, k) = C(n-1, k-1) + C(n-1, k)
#   C(n, 0) = C(n, n) = 1
#   C(n, k) = 0 при k < 0 или k > n
#
# Наивная рекурсия: экспоненциальное дерево с огромным числом повторов.
# Мемоизация / итерация: Θ(n·k) различных подзадач.
# ===========================================================================

def binom_naive(n: int, k: int) -> int:
    """
    Наивная рекурсия. Трудоёмкость Θ(C(n,k)·poly) ≈ экспонента по n.

    Каждый вызов порождает два, одни и те же (n',k') считаются многократно.
    """
    global calls
    calls += 1                          # считаем каждый вход в функцию
    if k < 0 or k > n:
        return 0                        # недопустимые аргументы
    if k == 0 or k == n:
        return 1                        # базовые случаи
    # C(n,k) = C(n-1, k-1) + C(n-1, k)
    return binom_naive(n - 1, k - 1) + binom_naive(n - 1, k)


def binom_memo(n: int, k: int, cache: Optional[Dict[Tuple[int, int], int]] = None) -> int:
    """
    Нисходящее ДП (мемоизация). Θ(n·k) различных состояний.

    Результат каждого (n,k) сохраняется в словаре; повторный запрос — O(1).
    """
    global calls
    calls += 1
    if cache is None:
        cache = {}                      # создаём кэш при первом вызове
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    key = (n, k)
    if key in cache:
        return cache[key]               # уже посчитано — берём из кэша
    # считаем и сохраняем
    cache[key] = binom_memo(n - 1, k - 1, cache) + binom_memo(n - 1, k, cache)
    return cache[key]


def binom_iter(n: int, k: int) -> int:
    """
    Итеративное вычисление через произведение:
      C(n,k) = Π_{i=1..k} (n-k+i) / i
    с k := min(k, n-k) для уменьшения числа шагов.
    Θ(k) умножений/делений, O(1) памяти.
    """
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)                   # симметрия C(n,k)=C(n,n-k)
    res = 1
    for i in range(1, k + 1):
        # умножаем на (n-k+i) и делим на i; целочисленно без переполнения дробей
        res = res * (n - k + i) // i
    return res


@functools.lru_cache(maxsize=None)
def binom_lru(n: int, k: int) -> int:
    """
    Та же рекурсия, но кэш — встроенный lru_cache (доп. Г6).

    Счётчик calls здесь не ведётся: декоратор перехватывает вызовы.
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    return binom_lru(n - 1, k - 1) + binom_lru(n - 1, k)


# ===========================================================================
# 2. ЗАДАЧА Б: МАКСИМАЛЬНАЯ СУММА ПОДОТРЕЗКА (Kadane / ДП)
# ===========================================================================
#
# D[i] = максимальная сумма подотрезка, оканчивающегося в i:
#   D[i] = max(a[i], D[i-1] + a[i])
# Ответ: max_i D[i].
# Восстановление: запоминаем границы текущего и лучшего отрезков.
# Трудоёмкость Θ(n), память O(1).
# ===========================================================================

def max_subarray(arr: List[int]) -> Tuple[int, int, int]:
    """
    Возвращает (сумма, left, right) — оптимальный подотрезок arr[left..right].

    Классический алгоритм Кадане с восстановлением границ.
    """
    best = cur = arr[0]                 # лучшая и текущая суммы
    best_l = best_r = cur_l = 0         # границы лучшего и текущего отрезков
    for i in range(1, len(arr)):
        # начинаем новый отрезок с i, если старый + a[i] хуже a[i]
        if cur + arr[i] < arr[i]:
            cur = arr[i]
            cur_l = i
        else:
            cur += arr[i]               # продлеваем текущий отрезок
        if cur > best:
            best = cur
            best_l, best_r = cur_l, i
    return best, best_l, best_r


# ===========================================================================
# 3. ИЗМЕРЕНИЯ
# ===========================================================================

def measure_binom() -> None:
    """Сравнение трёх реализаций C(n, n//2) + lru_cache."""
    global calls
    print('=' * 78)
    print('Задача А: биномиальный коэффициент C(n, k), k = n//2')
    print('=' * 78)
    hdr = (f"{'n':>4} {'k':>3} {'naive_s':>10} {'calls':>12}"
           f" {'memo_s':>10} {'calls':>8} {'iter_s':>10} {'lru_s':>10} {'C(n,k)':>12}")
    print(hdr)

    for n in [10, 15, 18, 20, 22, 24, 26, 28]:
        k = n // 2

        # --- наивная (только до n=22: дальше слишком долго) ---
        t_naive_s, c_naive_s = '—', '—'
        r_naive = None
        if n <= 22:
            calls = 0
            t0 = time.perf_counter()
            r_naive = binom_naive(n, k)
            t_naive = time.perf_counter() - t0
            t_naive_s = f'{t_naive:.4f}'
            c_naive_s = str(calls)

        # --- мемоизация ---
        calls = 0
        t0 = time.perf_counter()
        r_memo = binom_memo(n, k)
        t_memo = time.perf_counter() - t0
        c_memo = calls

        # --- итеративная ---
        t0 = time.perf_counter()
        r_iter = binom_iter(n, k)
        t_iter = time.perf_counter() - t0

        # --- lru_cache (Г6) ---
        binom_lru.cache_clear()         # очищаем кэш перед замером
        t0 = time.perf_counter()
        r_lru = binom_lru(n, k)
        t_lru = time.perf_counter() - t0

        assert r_memo == r_iter == r_lru
        if r_naive is not None:
            assert r_naive == r_memo

        print(f'{n:4d} {k:3d} {t_naive_s:>10} {c_naive_s:>12}'
              f' {t_memo:10.6f} {c_memo:8d} {t_iter:10.6f} {t_lru:10.6f} {r_iter:12d}')

    print('\nНаивная рекурсия при n>22 не измерялась (время >> 1 с).')
    print('Число вызовов memo ≈ 2·(число различных пар) — линейно по числу подзадач.')


def measure_max_subarray() -> None:
    """Задача Б: максимальная сумма подотрезка на 10 000 элементов."""
    print('\n' + '=' * 78)
    print('Задача Б: максимальная сумма подотрезка (n = 10 000)')
    print('=' * 78)
    rng = random.Random(42)
    arr = [rng.randint(-50, 100) for _ in range(10_000)]

    t0 = time.perf_counter()
    s, left, right = max_subarray(arr)
    elapsed = time.perf_counter() - t0

    print(f'Оптимальная сумма: {s}')
    print(f'Границы: [{left} .. {right}], длина = {right - left + 1}')
    print(f'Проверка sum(arr[L:R+1]) = {sum(arr[left:right + 1])}')
    print(f'Время: {elapsed:.6f} с')
    # фрагмент для отчёта
    preview = arr[left:min(left + 5, right + 1)]
    print(f'Начало отрезка (первые элементы): {preview} ...')


def demo_small_tree() -> None:
    """Дерево рекурсии C(5,2): показать повторяющиеся подзадачи."""
    global calls
    print('\n' + '=' * 78)
    print('Дерево рекурсии (иллюстрация): C(5, 2)')
    print('=' * 78)
    calls = 0
    r = binom_naive(5, 2)
    print(f'C(5,2) = {r}, число вызовов наивной = {calls}')
    calls = 0
    r2 = binom_memo(5, 2)
    print(f'C(5,2) memo = {r2}, число вызовов = {calls}')
    print('Повторяющиеся подзадачи: C(3,1), C(2,1), C(2,0) и др. считаются многократно.')


def g6_compare() -> None:
    """Г6: своя мемоизация vs functools.lru_cache."""
    global calls
    print('\n' + '=' * 78)
    print('Г6: своя мемоизация vs functools.lru_cache')
    print('=' * 78)
    for n in [20, 24, 28]:
        k = n // 2
        # своя
        calls = 0
        t0 = time.perf_counter()
        r1 = binom_memo(n, k)
        t1 = time.perf_counter() - t0
        c1 = calls
        # lru
        binom_lru.cache_clear()
        t0 = time.perf_counter()
        r2 = binom_lru(n, k)
        t2 = time.perf_counter() - t0
        info = binom_lru.cache_info()
        print(f'n={n}: memo={t1:.6f}s calls={c1} | lru={t2:.6f}s cache_info={info}')
        assert r1 == r2
    print('lru_cache реализован на C и хранит кэш эффективнее чистого dict в Python;')
    print('поэтому обычно чуть быстрее. Семантика та же: каждая подзадача — один раз.')


if __name__ == '__main__':
    demo_small_tree()
    measure_binom()
    g6_compare()
    measure_max_subarray()
