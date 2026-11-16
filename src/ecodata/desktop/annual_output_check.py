"""Check an annual-average SWAT+ output against its yearly companion locally."""
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from itertools import islice
from pathlib import PurePosixPath

from ecodata.interface.swat_results import output_header
from .scan import decode, local_member


def output_values(row, name, yearly):
    values=defaultdict(dict)
    with local_member(row) as path, path.open("rb") as stream:
        prefix=list(islice(stream,row["swat_output_header"]+2))
        _,columns,units,optional_tail=output_header([decode(line) for line in prefix])
        position=columns.index(name)
        if not units.get(name):
            raise ValueError("单位未声明")
        for raw in stream:
            tokens=decode(raw).split()
            if not tokens:
                continue
            if len(tokens)!=len(columns) and not (optional_tail and len(tokens)==len(columns)-2):
                raise ValueError("记录列数不符")
            identity=int(tokens[4]),int(tokens[5]),tokens[6]
            year=int(tokens[3]) if yearly else None
            value=Decimal(tokens[position].upper().replace("D","E"))
            if not value.is_finite() or abs(value)>=Decimal("1e30") or year in values[identity]:
                raise ValueError("重复年份或无效数值")
            values[identity][year]=value
    return values


def compare(rows,fields):
    parts={row.get("declared_output_period"):(row,field) for row,field in zip(rows,fields)}
    if set(parts)!={"average_annual","yearly"}:
        return dict(status="unavailable",basis="annual_and_yearly_pair_required")
    average,average_field=parts["average_annual"]
    yearly,year_field=parts["yearly"]
    paths=[PurePosixPath(row["name"]) for row in (average,yearly)]
    if (paths[0].parent!=paths[1].parent or paths[0].stem[:-3]!=paths[1].stem[:-3]
            or ((average["ref"][1] or yearly["ref"][1]) and average["ref"][0]!=yearly["ref"][0])
            or average_field["name"]!=year_field["name"]):
        return dict(status="unavailable",basis="same_scenario_and_output_variable_unconfirmed")
    if not average_field["unit"] or average_field["unit"]!=year_field["unit"]:
        return dict(status="unavailable",basis="annual_unit_not_confirmed")
    period=average.get("declared_output_print_period")
    if not period or not average.get("single_average_annual_window"):
        return dict(status="unavailable",basis="average_annual_window_unconfirmed")
    try:
        averaged=output_values(average,average_field["name"],False)
        observations=output_values(yearly,year_field["name"],True)
    except (OSError,ValueError,InvalidOperation,RuntimeError,KeyError,IndexError):
        return dict(status="unavailable",basis="native_yearly_output_unreadable")
    years=set(range(period["start_year"],period["end_year"]+1))
    common=averaged.keys() & observations.keys()
    complete={identity for identity in common if years<=observations[identity].keys()}
    agreements=0
    for identity in complete:
        target=averaged[identity][None]
        yearly_values=[observations[identity][year] for year in years]
        mean=sum(yearly_values)/len(years)
        # Both published values are rounded to their printed precision.
        quantum=lambda value: Decimal(1).scaleb(value.as_tuple().exponent)
        tolerance=(quantum(target)+sum(map(quantum,yearly_values))/len(years))/2
        agreements+=abs(target-mean)<=tolerance
    return dict(status="checked",basis="swat_yearly_mean_comparison",year_window=period,
                matched_objects=len(common),compared_objects=len(complete),matching_means=agreements,
                annual_objects=len(averaged),yearly_objects=len(observations),
                years_required=len(years))
