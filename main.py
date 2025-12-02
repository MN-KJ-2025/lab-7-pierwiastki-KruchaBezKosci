# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import numpy.polynomial.polynomial as nppoly


def roots_20(coef: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja wyznaczająca miejsca zerowe wielomianu funkcją
    `nppoly.polyroots()`, najpierw lekko zaburzając wejściowe współczynniki 
    wielomianu (N(0,1) * 1e-10).

    Args:
        coef (np.ndarray): Wektor współczynników wielomianu (n,).

    Returns:
        (tuple[np.ndarray, np. ndarray]):
            - Zaburzony wektor współczynników (n,),
            - Wektor miejsc zerowych (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(coef,np.ndarray):
        print("BŁĄD: To nie jest tablica numpy (isinstance zwróciło False)")
        return None
    if coef.size == 0:
        print("BŁĄD: Tablica jest pusta (len == 0)")
        return None
    if coef.ndim != 1:
        print(f"BŁĄD: Zła liczba wymiarów. Oczekiwano 1, otrzymano: {coef.ndim}")
        print(f"INFO: Kształt tablicy to: {coef.shape}")
        return None
        
    coef_zab = coef + (np.random.random_sample(len(coef)) * 1e-10)
    return (np.array(coef_zab),np.array(nppoly.polyroots(coef_zab)))


def frob_a(coef: np.ndarray) -> np.ndarray | None:
    """Funkcja służąca do wyznaczenia macierzy Frobeniusa na podstawie
    współczynników jej wielomianu charakterystycznego:
    w(x) = a_n*x^n + a_{n-1}*x^{n-1} + ... + a_2*x^2 + a_1*x + a_0

    Testy wymagają poniższej definicji macierzy Frobeniusa (implementacja dla 
    innych postaci nie jest zabroniona):
    F = [[       0,        1,        0,   ...,            0],
         [       0,        0,        1,   ...,            0],
         [       0,        0,        0,   ...,            0],
         [     ...,      ...,      ...,   ...,          ...],
         [-a_0/a_n, -a_1/a_n, -a_2/a_n,   ..., -a_{n-1}/a_n]]

    Args:
        coef (np.narray): Wektor współczynników wielomianu (n,).

    Returns:
        (np.ndarray): Macierz Frobeniusa o rozmiarze (n,n).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(coef,np.ndarray):
        return None
    a_n = coef[-1]
    
    if a_n == 0:
        return None
    
    n = coef.size - 1
    matrix = np.zeros((n, n), dtype=float)
    
    for i in range(n - 1):
        matrix[i, i + 1] = 1.0

    matrix[-1, :] = -coef[:-1] / a_n
    return matrix



def is_nonsingular(A: np.ndarray) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz NIE JEST singularna. Przy
    implementacji należy pamiętać o definicji zera maszynowego.

    Args:
        A (np.ndarray): Macierz (n,n) do przetestowania.

    Returns:
        (bool): `True`, jeżeli macierz A nie jest singularna, w przeciwnym 
            wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, np.ndarray):
        return None
    
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.size == 0:
        return None

    epsilon = np.finfo(float).eps
  
    det = np.linalg.det(A)

    return np.abs(det) > epsilon

