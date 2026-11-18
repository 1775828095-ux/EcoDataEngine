"""Check an annual-average SWAT+ output against its yearly companion locally."""
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from itertools import islice
from pathlib import PurePosixPath

from ecodata.interface.swat_results import output_header
from ecodata.io.record_layout import RecordLayoutError,split_declared_record
from .scan import decode, local_member


class OutputReadError(ValueError):
    def __init__(self, code, line=None):
        super().__init__(code)
        self.code=code
        self.line=line


def output_values(row, name, yearly, diagnostics=None):
    values=defaultdict(dict)
    with local_member(row) as path, path.open("rb") as stream:
        prefix=list(islice(stream,row["swat_output_header"]+2))
        _,columns,units,optional_tail=output_header([decode(line) for line in prefix])
        position=columns.index(name)
        numeric_positions=[columns.index(column) for column in units]
        if not units.get(name):
            raise ValueError("单位未声明")
        for line,raw in enumerate(stream,len(prefix)+1):
            tokens=decode(raw).split()
            if not tokens:
                continue
            try:
                tokens,extra=split_declared_record(tokens,len(columns),numeric_columns=numeric_positions,
                    integer_columns=range(6),optional_tail=2 if optional_tail else 0)
            except RecordLayoutError as error:
                raise OutputReadError(error.code,line) from error
            if extra and diagnostics is not None:
                diagnostics.append(dict(line=line,extra_tokens=len(extra),basis="trailing_text_annotation"))
            try:
                for token in tokens[:6]:
                    int(token)
                # Validate the declared numeric prefix before accepting any trailing annotation.
                for offset in numeric_positions:
                    number=Decimal(tokens[offset].upper().replace("D","E"))
                    if not number.is_finite() or abs(number)>=Decimal("1e30"):
                        raise OutputReadError("invalid_numeric_value",line)
                identity=int(tokens[4]),int(tokens[5]),tokens[6]
                year=int(tokens[3]) if yearly else None
                value=Decimal(tokens[position].upper().replace("D","E"))
            except (ValueError,InvalidOperation,IndexError) as error:
                if isinstance(error,OutputReadError):
                    raise
                raise OutputReadError("invalid_identity_or_numeric_prefix",line) from error
            if not value.is_finite() or abs(value)>=Decimal("1e30") or year in values[identity]:
                raise OutputReadError("duplicate_year_or_invalid_value",line)
            values[identity][year]=value
    if not values:
        raise OutputReadError("no_output_records")
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
    outputs=[]
    annotations=[]
    for row,field,annual in ((average,average_field,False),(yearly,year_field,True)):
        notes=[]
        try:
            outputs.append(output_values(row,field["name"],annual,notes))
        except (OSError,ValueError,InvalidOperation,RuntimeError,KeyError,IndexError) as error:
            return dict(status="unavailable",basis="native_yearly_output_unreadable",
                        failed_output="yearly" if annual else "average_annual",
                        read_error=getattr(error,"code","native_output_reader_error"),
                        line=getattr(error,"line",None))
        annotations.extend(dict(output="yearly" if annual else "average_annual",**note) for note in notes)
    averaged,observations=outputs
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
                years_required=len(years),trailing_annotations=annotations)
