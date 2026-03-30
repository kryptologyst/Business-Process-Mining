"""Tests for event log functionality."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from business_process_mining.data.event_log import EventLog, EventLogGenerator, Event


class TestEvent:
    """Test Event class."""
    
    def test_event_creation(self):
        """Test event creation."""
        event = Event(
            case_id="Case_001",
            activity="Start",
            timestamp=datetime(2023, 1, 1, 9, 0, 0),
            resource="System",
            cost=0.0,
            duration=0.1
        )
        
        assert event.case_id == "Case_001"
        assert event.activity == "Start"
        assert event.resource == "System"
        assert event.cost == 0.0
        assert event.duration == 0.1
    
    def test_event_optional_fields(self):
        """Test event with optional fields."""
        event = Event(
            case_id="Case_001",
            activity="Start",
            timestamp=datetime(2023, 1, 1, 9, 0, 0)
        )
        
        assert event.case_id == "Case_001"
        assert event.activity == "Start"
        assert event.resource is None
        assert event.cost is None
        assert event.duration is None


class TestEventLog:
    """Test EventLog class."""
    
    def test_event_log_creation(self):
        """Test event log creation."""
        events = [
            Event(
                case_id="Case_001",
                activity="Start",
                timestamp=datetime(2023, 1, 1, 9, 0, 0)
            ),
            Event(
                case_id="Case_001",
                activity="End",
                timestamp=datetime(2023, 1, 1, 10, 0, 0)
            )
        ]
        
        event_log = EventLog(events)
        
        assert len(event_log.events) == 2
        assert len(event_log.case_ids) == 1
        assert "Case_001" in event_log.case_ids
    
    def test_dataframe_property(self):
        """Test dataframe property."""
        events = [
            Event(
                case_id="Case_001",
                activity="Start",
                timestamp=datetime(2023, 1, 1, 9, 0, 0)
            ),
            Event(
                case_id="Case_001",
                activity="End",
                timestamp=datetime(2023, 1, 1, 10, 0, 0)
            )
        ]
        
        event_log = EventLog(events)
        df = event_log.dataframe
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['case_id', 'activity', 'timestamp', 'resource', 'cost', 'duration']
    
    def test_get_case_trace(self):
        """Test getting case trace."""
        events = [
            Event(
                case_id="Case_001",
                activity="Start",
                timestamp=datetime(2023, 1, 1, 9, 0, 0)
            ),
            Event(
                case_id="Case_001",
                activity="Process",
                timestamp=datetime(2023, 1, 1, 9, 30, 0)
            ),
            Event(
                case_id="Case_001",
                activity="End",
                timestamp=datetime(2023, 1, 1, 10, 0, 0)
            )
        ]
        
        event_log = EventLog(events)
        trace = event_log.get_case_trace("Case_001")
        
        assert trace == ["Start", "Process", "End"]
    
    def test_get_case_duration(self):
        """Test getting case duration."""
        events = [
            Event(
                case_id="Case_001",
                activity="Start",
                timestamp=datetime(2023, 1, 1, 9, 0, 0)
            ),
            Event(
                case_id="Case_001",
                activity="End",
                timestamp=datetime(2023, 1, 1, 11, 0, 0)
            )
        ]
        
        event_log = EventLog(events)
        duration = event_log.get_case_duration("Case_001")
        
        assert duration == 2.0  # 2 hours
    
    def test_filter_by_activity(self):
        """Test filtering by activity."""
        events = [
            Event(
                case_id="Case_001",
                activity="Start",
                timestamp=datetime(2023, 1, 1, 9, 0, 0)
            ),
            Event(
                case_id="Case_001",
                activity="Process",
                timestamp=datetime(2023, 1, 1, 9, 30, 0)
            ),
            Event(
                case_id="Case_002",
                activity="Start",
                timestamp=datetime(2023, 1, 2, 9, 0, 0)
            )
        ]
        
        event_log = EventLog(events)
        filtered_log = event_log.filter_by_activity("Start")
        
        assert len(filtered_log.events) == 2
        assert all(event.activity == "Start" for event in filtered_log.events)
    
    def test_to_pm4py_format(self):
        """Test conversion to PM4Py format."""
        events = [
            Event(
                case_id="Case_001",
                activity="Start",
                timestamp=datetime(2023, 1, 1, 9, 0, 0),
                resource="System"
            )
        ]
        
        event_log = EventLog(events)
        pm4py_df = event_log.to_pm4py_format()
        
        assert isinstance(pm4py_df, pd.DataFrame)
        assert 'case:concept:name' in pm4py_df.columns
        assert 'concept:name' in pm4py_df.columns
        assert 'time:timestamp' in pm4py_df.columns
        assert 'org:resource' in pm4py_df.columns


class TestEventLogGenerator:
    """Test EventLogGenerator class."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = EventLogGenerator(seed=42)
        assert generator.seed == 42
    
    def test_generate_simple_approval_process(self):
        """Test simple approval process generation."""
        generator = EventLogGenerator(seed=42)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        event_log = generator.generate_simple_approval_process(
            num_cases=10,
            start_date=start_date,
            end_date=end_date
        )
        
        assert len(event_log.case_ids) == 10
        assert len(event_log.events) > 0
        
        # Check that all cases have Start and End activities
        for case_id in event_log.case_ids:
            trace = event_log.get_case_trace(case_id)
            assert "Start" in trace
            assert "End" in trace
    
    def test_generate_complex_loan_process(self):
        """Test complex loan process generation."""
        generator = EventLogGenerator(seed=42)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        event_log = generator.generate_complex_loan_process(
            num_cases=10,
            start_date=start_date,
            end_date=end_date
        )
        
        assert len(event_log.case_ids) == 10
        assert len(event_log.events) > 0
        
        # Check that all cases have Application activity
        for case_id in event_log.case_ids:
            trace = event_log.get_case_trace(case_id)
            assert "Application" in trace
    
    def test_generate_manufacturing_process(self):
        """Test manufacturing process generation."""
        generator = EventLogGenerator(seed=42)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        event_log = generator.generate_manufacturing_process(
            num_cases=10,
            start_date=start_date,
            end_date=end_date
        )
        
        assert len(event_log.case_ids) == 10
        assert len(event_log.events) > 0
        
        # Check that all cases have Order and Ship activities
        for case_id in event_log.case_ids:
            trace = event_log.get_case_trace(case_id)
            assert "Order" in trace
            assert "Ship" in trace
    
    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        generator1 = EventLogGenerator(seed=42)
        generator2 = EventLogGenerator(seed=42)
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        log1 = generator1.generate_simple_approval_process(10, start_date, end_date)
        log2 = generator2.generate_simple_approval_process(10, start_date, end_date)
        
        # Should generate identical logs
        assert len(log1.events) == len(log2.events)
        assert log1.case_ids == log2.case_ids
