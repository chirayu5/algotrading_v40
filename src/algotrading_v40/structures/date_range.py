import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class DateRange:
  start_date: datetime.date
  end_date: datetime.date

  def __post_init__(self):
    if not isinstance(self.start_date, datetime.date):
      raise ValueError("start_date must be a datetime.date")
    if not isinstance(self.end_date, datetime.date):
      raise ValueError("end_date must be a datetime.date")
    if self.start_date > self.end_date:
      raise ValueError("start_date must be before end_date")

  def slide(self, days: int) -> "DateRange":
    return DateRange(
      self.start_date + datetime.timedelta(days=days),
      self.end_date + datetime.timedelta(days=days),
    )

  def n_days(self) -> int:
    # both start and end dates are inclusive
    return (self.end_date - self.start_date).days + 1
