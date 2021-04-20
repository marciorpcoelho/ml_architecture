from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import cvxpy as cp
import os
from py_dotenv import read_dotenv
dotenv_path = '/info.env'
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
read_dotenv(base_path + dotenv_path)

app = FastAPI(
    title='Predictive Analytics API Endpoint',
    description='API endpoint for all optimization/prediction projects for the Predictive Analytics Teams.',
    version='0.1'
)

api_endpoint_ip = os.getenv('API_IP')


class OptimizationDataIn(BaseModel):
    name: Optional[str] = None
    df_solve: str
    sel_local: str
    group: str
    goal_value: int
    goal_type: str
    non_goal_type: str
    dtss_goal: int
    max_part_number: int
    minimum_cost_or_pvp: float
    sel_metric: str

    class Config:
        arbitrary_types_allowed = True


class OptimizationDataHyundaiHondaIn(BaseModel):
    df_solve: str
    parameter_restriction_vectors: list
    sel_order_size: int


class OptimizationDataHyundaiHondaOut(BaseModel):
    selection: list
    status: str
    unique_ids: list
    optimization_total_sum: float


class OptimizationDataOut(BaseModel):
    selection: list
    unique_parts: list
    descriptions: list
    values: list
    other_values: list
    dtss: list
    above_goal_flag: int
    optimization_total_sum: int


class OptimizationDataBavieraOut(BaseModel):
    selection: list
    status: str
    optimization_total_sum: float


@app.get("/optimizations/apv_parts_baviera/", response_model=OptimizationDataOut)
def solver(item: OptimizationDataIn):
    df_solve = pd.read_json(item.df_solve)

    above_goal_flag = 0
    df_solve = df_solve[df_solve[item.sel_metric] <= item.dtss_goal]

    if item.minimum_cost_or_pvp:
        df_solve = df_solve[df_solve[item.goal_type] >= item.minimum_cost_or_pvp]

    unique_parts = df_solve['Part_Ref'].unique()
    descriptions = [x for x in df_solve['Part_Desc']]
    df_solve = df_solve[df_solve['Part_Ref'].isin(unique_parts)]

    n_size = df_solve['Part_Ref'].nunique()  # Number of different parts
    if not n_size:
        return None

    values = np.array(df_solve[item.goal_type].values.tolist())  # Costs/Sale prices for each reference, info#1
    other_values = df_solve[item.non_goal_type].values.tolist()
    dtss = np.array(df_solve[item.sel_metric].values.tolist())  # Days to Sell of each reference, info#2

    selection = cp.Variable(n_size, integer=True)

    dtss_constraint = cp.multiply(selection.T, dtss)

    total_value = selection @ values  # Changed in CVXPY 1.1

    if item.max_part_number:
        problem_testing_2 = cp.Problem(cp.Maximize(total_value), [dtss_constraint <= item.dtss_goal, selection >= 0, selection <= 100, cp.sum(selection) <= item.max_part_number])
    else:
        problem_testing_2 = cp.Problem(cp.Maximize(total_value), [dtss_constraint <= item.dtss_goal, selection >= 0, selection <= 100])

    result = problem_testing_2.solve(solver=cp.GLPK_MI, verbose=False, parallel=True)

    if selection.value is not None:
        if result >= item.goal_value:
            above_goal_flag = 1

    response = {
        'selection': [qty for qty in selection.value],
        'unique_parts': [part for part in unique_parts],
        'descriptions': [desc for desc in descriptions],
        'values': [value for value in values],
        'other_values': [value for value in other_values],
        'dtss': [dts for dts in dtss],
        'above_goal_flag': above_goal_flag,
        'optimization_total_sum': result
    }

    return response


@app.get("/optimizations/vhe_hyundai_honda/", response_model=OptimizationDataHyundaiHondaOut)
def solver(item: OptimizationDataHyundaiHondaIn):
    dataset = pd.read_json(item.df_solve)
    parameter_restriction = item.parameter_restriction_vectors

    unique_ids_count = dataset['ML_VehicleData_Code'].nunique()
    unique_ids = dataset['ML_VehicleData_Code'].unique()

    scores_values = [dataset[dataset['ML_VehicleData_Code'] == x]['Average_Score_Euros'].head(1).values[0] for x in unique_ids]  # uniques() command doesn't work as intended because there are configurations (Configuration IDs) with repeated average score

    selection = cp.Variable(unique_ids_count, integer=True)

    order_size_restriction = cp.sum(selection) <= item.sel_order_size
    total_value = selection @ scores_values  # Changed in CVXPY 1.1

    problem = cp.Problem(cp.Maximize(total_value), [selection >= 0, selection <= 100,
                                                    order_size_restriction,
                                                    ] + parameter_restriction)

    result = problem.solve(solver=cp.GLPK_MI, verbose=False, qcp=True)

    response = {
        'selection': [qty for qty in selection.value],
        'status': problem.status,
        'unique_ids': [int(ref) for ref in unique_ids],  # json does not recognize numpy data types (int64 in this case) and so the values inside this list need to be converted to int before being serialized;
        'optimization_total_sum': result,
    }

    return response


@app.get("/optimizations/vhe_baviera/", response_model=OptimizationDataBavieraOut)
def solver(item: OptimizationDataHyundaiHondaIn):
    dataset = pd.read_json(item.df_solve)
    parameter_restriction_vectors = item.parameter_restriction_vectors
    parameter_restriction = []

    unique_ids_count = dataset['Configuration_ID'].nunique()
    scores = dataset['Score (€)'].values.tolist()

    selection = cp.Variable(unique_ids_count, integer=True)
    for parameter_vector in parameter_restriction_vectors:
        parameter_restriction.append(selection >= parameter_vector)

    order_size_restriction = cp.sum(selection) <= item.sel_order_size
    total_value = selection * scores

    problem = cp.Problem(cp.Maximize(total_value), [selection >= 0, selection <= 100,
                                                    order_size_restriction,
                                                    ] + parameter_restriction)

    result = problem.solve(solver=cp.GLPK_MI, verbose=False, qcp=True)

    response = {
        'selection': [qty for qty in selection.value],
        'status': problem.status,
        'optimization_total_sum': result
    }

    return response

