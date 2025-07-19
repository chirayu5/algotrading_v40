import datetime

import pytest

from algotrading_v40.structures.date_range import DateRange


class TestDateRange:
  def test_valid_date_range(self):
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    date_range = DateRange(start_date, end_date)

    assert date_range.start_date == start_date
    assert date_range.end_date == end_date

  def test_same_start_and_end_date(self):
    date = datetime.date(2023, 1, 1)
    date_range = DateRange(date, date)

    assert date_range.start_date == date
    assert date_range.end_date == date
    assert date_range.n_days() == 1

  def test_invalid_start_date_type(self):
    with pytest.raises(ValueError, match="start_date must be a datetime.date"):
      DateRange("2023-01-01", datetime.date(2023, 1, 31))

  def test_invalid_end_date_type(self):
    with pytest.raises(ValueError, match="end_date must be a datetime.date"):
      DateRange(datetime.date(2023, 1, 1), "2023-01-31")

  def test_start_date_after_end_date(self):
    start_date = datetime.date(2023, 1, 31)
    end_date = datetime.date(2023, 1, 1)

    with pytest.raises(ValueError, match="start_date must be before end_date"):
      DateRange(start_date, end_date)

  def test_slide_positive_days(self):
    date_range = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    slid_range = date_range.slide(10)

    assert slid_range.start_date == datetime.date(2023, 1, 11)
    assert slid_range.end_date == datetime.date(2023, 2, 10)
    assert slid_range.n_days() == date_range.n_days() == 31

  def test_slide_negative_days(self):
    date_range = DateRange(datetime.date(2023, 1, 15), datetime.date(2023, 1, 31))
    slid_range = date_range.slide(-10)

    assert slid_range.start_date == datetime.date(2023, 1, 5)
    assert slid_range.end_date == datetime.date(2023, 1, 21)
    assert slid_range.n_days() == date_range.n_days() == 17

  def test_slide_zero_days(self):
    date_range = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    slid_range = date_range.slide(0)

    assert slid_range.start_date == date_range.start_date
    assert slid_range.end_date == date_range.end_date
    assert slid_range == date_range

  def test_n_days_across_years(self):
    start_date = datetime.date(2022, 12, 30)
    end_date = datetime.date(2023, 1, 2)
    date_range = DateRange(start_date, end_date)

    assert date_range.n_days() == 4

  def test_frozen_dataclass(self):
    date_range = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))

    with pytest.raises(AttributeError):
      date_range.start_date = datetime.date(2023, 2, 1)

    with pytest.raises(AttributeError):
      date_range.end_date = datetime.date(2023, 2, 28)
