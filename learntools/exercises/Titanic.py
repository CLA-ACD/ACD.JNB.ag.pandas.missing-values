import pandas as pd
from scipy import stats
import numpy as np
np.random.seed(0)

from learntools.core import *

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", index_col=0)
data = np.random.exponential(size=1000)
normalized = stats.boxcox(data)
fechas = {'dates': ['23-01-1957','28-02-1963','17-01-1970','02-01-1984','02-10-1989','07-01-1972','18-06-1990','10-11-1987','18-02-1969','01-05-1972','17-08-1984','02-08-1987']}
df_fechas = pd.DataFrame(data=fechas)

class Ejercicio1(EqualityCheckProblem):
    _var = 'p1'
    _expected = (
            df.isnull().sum(),
    )

class Ejercicio2(EqualityCheckProblem):
    _var = 'p2'
    _expected = (
            np.prod(df.shape),
    )

class Ejercicio3(EqualityCheckProblem):
    _var = 'p3'
    _expected = (
            df.isnull().sum().sum(),
    )

class Ejercicio4(EqualityCheckProblem):
    _var = 'p4'
    _expected = (
            (df.isnull().sum().sum() / np.prod(df.shape)) * 100,
    )

class Ejercicio5(EqualityCheckProblem):
    _var = 'p5'
    _expected = (
            df.dropna(),
    )

class Ejercicio6(EqualityCheckProblem):
    _var = 'p6'
    _expected = (
            df.dropna(axis=1),
    )

class Ejercicio7(EqualityCheckProblem):
    _var = 'p7'
    _expected = (
            df.fillna(0),
    )

class Ejercicio8(EqualityCheckProblem):
    _var = 'p8'
    _expected = (
            data,
    )

class Ejercicio9(CodingProblem):
    _var = 'p9'
    
    def check(self, solution):
        assert np.all(solution[0] == normalized[0]) & np.all(solution[1] == normalized[1])

class Ejercicio10(EqualityCheckProblem):
    _var = 'p10'
    _expected = (
            pd.to_datetime(df_fechas['dates'], format='%d-%m-%Y'),
    )


qvars = bind_exercises(globals(), [
    Ejercicio1,
    Ejercicio2,
    Ejercicio3,
    Ejercicio4,
    Ejercicio5,
    Ejercicio6,
    Ejercicio7,
    Ejercicio8,
    Ejercicio9,
    Ejercicio10,
    ],
    var_format='q{n}',
    )
__all__ = list(qvars)