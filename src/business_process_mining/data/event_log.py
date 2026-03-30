"""Event log data structures and generation utilities."""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator


class Event(BaseModel):
    """Represents a single event in a business process."""
    
    case_id: str = Field(..., description="Unique identifier for the process instance")
    activity: str = Field(..., description="Name of the activity performed")
    timestamp: datetime = Field(..., description="When the activity occurred")
    resource: Optional[str] = Field(None, description="Resource (person/system) that performed the activity")
    cost: Optional[float] = Field(None, description="Cost associated with this activity")
    duration: Optional[float] = Field(None, description="Duration of the activity in hours")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EventLog:
    """Represents a collection of events forming a business process log."""
    
    def __init__(self, events: List[Event]) -> None:
        """Initialize event log with a list of events.
        
        Args:
            events: List of Event objects
        """
        self.events = events
        self._df: Optional[pd.DataFrame] = None
    
    @property
    def dataframe(self) -> pd.DataFrame:
        """Get event log as pandas DataFrame.
        
        Returns:
            DataFrame with events sorted by case_id and timestamp
        """
        if self._df is None:
            data = []
            for event in self.events:
                data.append({
                    'case_id': event.case_id,
                    'activity': event.activity,
                    'timestamp': event.timestamp,
                    'resource': event.resource,
                    'cost': event.cost,
                    'duration': event.duration
                })
            self._df = pd.DataFrame(data)
            self._df = self._df.sort_values(['case_id', 'timestamp']).reset_index(drop=True)
        return self._df
    
    @property
    def case_ids(self) -> List[str]:
        """Get unique case IDs in the event log.
        
        Returns:
            List of unique case IDs
        """
        return sorted(list(set(event.case_id for event in self.events)))
    
    @property
    def activities(self) -> List[str]:
        """Get unique activities in the event log.
        
        Returns:
            List of unique activities
        """
        return sorted(list(set(event.activity for event in self.events)))
    
    @property
    def resources(self) -> List[str]:
        """Get unique resources in the event log.
        
        Returns:
            List of unique resources (excluding None)
        """
        resources = set(event.resource for event in self.events if event.resource is not None)
        return sorted(list(resources))
    
    def get_case_trace(self, case_id: str) -> List[str]:
        """Get the sequence of activities for a specific case.
        
        Args:
            case_id: The case identifier
            
        Returns:
            List of activities in chronological order
        """
        case_events = [event for event in self.events if event.case_id == case_id]
        case_events.sort(key=lambda x: x.timestamp)
        return [event.activity for event in case_events]
    
    def get_case_duration(self, case_id: str) -> Optional[float]:
        """Get the total duration of a case in hours.
        
        Args:
            case_id: The case identifier
            
        Returns:
            Duration in hours, or None if timestamps are missing
        """
        case_events = [event for event in self.events if event.case_id == case_id]
        if len(case_events) < 2:
            return None
        
        case_events.sort(key=lambda x: x.timestamp)
        start_time = case_events[0].timestamp
        end_time = case_events[-1].timestamp
        return (end_time - start_time).total_seconds() / 3600
    
    def filter_by_activity(self, activity: str) -> "EventLog":
        """Filter events by activity name.
        
        Args:
            activity: Activity name to filter by
            
        Returns:
            New EventLog with filtered events
        """
        filtered_events = [event for event in self.events if event.activity == activity]
        return EventLog(filtered_events)
    
    def filter_by_time_range(self, start_time: datetime, end_time: datetime) -> "EventLog":
        """Filter events by time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            New EventLog with filtered events
        """
        filtered_events = [
            event for event in self.events 
            if start_time <= event.timestamp <= end_time
        ]
        return EventLog(filtered_events)
    
    def to_pm4py_format(self) -> pd.DataFrame:
        """Convert to PM4Py-compatible format.
        
        Returns:
            DataFrame in PM4Py event log format
        """
        df = self.dataframe.copy()
        df = df.rename(columns={
            'case_id': 'case:concept:name',
            'activity': 'concept:name',
            'timestamp': 'time:timestamp',
            'resource': 'org:resource'
        })
        return df


class EventLogGenerator:
    """Generates synthetic event logs for testing and demonstration."""
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize the generator with a random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_simple_approval_process(
        self, 
        num_cases: int = 100,
        start_date: datetime = datetime(2023, 1, 1),
        end_date: datetime = datetime(2023, 12, 31)
    ) -> EventLog:
        """Generate a simple approval process event log.
        
        Process flow: Start -> Check Form -> [Approve|Reject] -> End
        
        Args:
            num_cases: Number of process instances to generate
            start_date: Start date for event generation
            end_date: End date for event generation
            
        Returns:
            Generated EventLog
        """
        events = []
        time_range = (end_date - start_date).days
        
        for case_id in range(1, num_cases + 1):
            case_start = start_date + timedelta(days=random.randint(0, time_range))
            
            # Start event
            events.append(Event(
                case_id=f"Case_{case_id:04d}",
                activity="Start",
                timestamp=case_start,
                resource="System",
                cost=0.0,
                duration=0.1
            ))
            
            # Check Form event
            check_time = case_start + timedelta(hours=random.uniform(1, 24))
            events.append(Event(
                case_id=f"Case_{case_id:04d}",
                activity="Check Form",
                timestamp=check_time,
                resource=random.choice(["Clerk_A", "Clerk_B", "Clerk_C"]),
                cost=random.uniform(10, 50),
                duration=random.uniform(0.5, 2.0)
            ))
            
            # Decision point - 80% approval rate
            decision_time = check_time + timedelta(hours=random.uniform(2, 48))
            if random.random() < 0.8:
                # Approve
                events.append(Event(
                    case_id=f"Case_{case_id:04d}",
                    activity="Approve",
                    timestamp=decision_time,
                    resource=random.choice(["Manager_A", "Manager_B"]),
                    cost=random.uniform(25, 75),
                    duration=random.uniform(0.2, 1.0)
                ))
                
                # End (approved)
                end_time = decision_time + timedelta(hours=random.uniform(1, 12))
                events.append(Event(
                    case_id=f"Case_{case_id:04d}",
                    activity="End",
                    timestamp=end_time,
                    resource="System",
                    cost=0.0,
                    duration=0.1
                ))
            else:
                # Reject
                events.append(Event(
                    case_id=f"Case_{case_id:04d}",
                    activity="Reject",
                    timestamp=decision_time,
                    resource=random.choice(["Manager_A", "Manager_B"]),
                    cost=random.uniform(15, 40),
                    duration=random.uniform(0.2, 0.5)
                ))
                
                # End (rejected)
                end_time = decision_time + timedelta(hours=random.uniform(1, 6))
                events.append(Event(
                    case_id=f"Case_{case_id:04d}",
                    activity="End",
                    timestamp=end_time,
                    resource="System",
                    cost=0.0,
                    duration=0.1
                ))
        
        return EventLog(events)
    
    def generate_complex_loan_process(
        self,
        num_cases: int = 200,
        start_date: datetime = datetime(2023, 1, 1),
        end_date: datetime = datetime(2023, 12, 31)
    ) -> EventLog:
        """Generate a complex loan approval process event log.
        
        Process flow: Application -> Credit Check -> [Auto-approve|Manual Review] -> 
        [Approve|Reject] -> [Disburse|End]
        
        Args:
            num_cases: Number of process instances to generate
            start_date: Start date for event generation
            end_date: End date for event generation
            
        Returns:
            Generated EventLog
        """
        events = []
        time_range = (end_date - start_date).days
        
        for case_id in range(1, num_cases + 1):
            case_start = start_date + timedelta(days=random.randint(0, time_range))
            
            # Application
            events.append(Event(
                case_id=f"Loan_{case_id:04d}",
                activity="Application",
                timestamp=case_start,
                resource="Customer",
                cost=0.0,
                duration=random.uniform(0.5, 2.0)
            ))
            
            # Credit Check
            credit_time = case_start + timedelta(hours=random.uniform(2, 24))
            events.append(Event(
                case_id=f"Loan_{case_id:04d}",
                activity="Credit Check",
                timestamp=credit_time,
                resource="Credit_System",
                cost=random.uniform(5, 15),
                duration=random.uniform(0.1, 0.5)
            ))
            
            # Decision point - 60% auto-approve
            decision_time = credit_time + timedelta(hours=random.uniform(1, 12))
            if random.random() < 0.6:
                # Auto-approve
                events.append(Event(
                    case_id=f"Loan_{case_id:04d}",
                    activity="Auto-approve",
                    timestamp=decision_time,
                    resource="Credit_System",
                    cost=random.uniform(2, 8),
                    duration=0.1
                ))
                
                # Disburse
                disburse_time = decision_time + timedelta(hours=random.uniform(1, 48))
                events.append(Event(
                    case_id=f"Loan_{case_id:04d}",
                    activity="Disburse",
                    timestamp=disburse_time,
                    resource="Banking_System",
                    cost=random.uniform(10, 30),
                    duration=random.uniform(0.5, 2.0)
                ))
            else:
                # Manual Review
                review_time = decision_time + timedelta(hours=random.uniform(24, 72))
                events.append(Event(
                    case_id=f"Loan_{case_id:04d}",
                    activity="Manual Review",
                    timestamp=review_time,
                    resource=random.choice(["Loan_Officer_A", "Loan_Officer_B"]),
                    cost=random.uniform(50, 150),
                    duration=random.uniform(1.0, 4.0)
                ))
                
                # Final decision - 70% approve after manual review
                final_time = review_time + timedelta(hours=random.uniform(2, 24))
                if random.random() < 0.7:
                    # Approve
                    events.append(Event(
                        case_id=f"Loan_{case_id:04d}",
                        activity="Approve",
                        timestamp=final_time,
                        resource=random.choice(["Loan_Officer_A", "Loan_Officer_B"]),
                        cost=random.uniform(25, 75),
                        duration=random.uniform(0.5, 1.5)
                    ))
                    
                    # Disburse
                    disburse_time = final_time + timedelta(hours=random.uniform(1, 48))
                    events.append(Event(
                        case_id=f"Loan_{case_id:04d}",
                        activity="Disburse",
                        timestamp=disburse_time,
                        resource="Banking_System",
                        cost=random.uniform(10, 30),
                        duration=random.uniform(0.5, 2.0)
                    ))
                else:
                    # Reject
                    events.append(Event(
                        case_id=f"Loan_{case_id:04d}",
                        activity="Reject",
                        timestamp=final_time,
                        resource=random.choice(["Loan_Officer_A", "Loan_Officer_B"]),
                        cost=random.uniform(15, 40),
                        duration=random.uniform(0.2, 0.8)
                    ))
        
        return EventLog(events)
    
    def generate_manufacturing_process(
        self,
        num_cases: int = 150,
        start_date: datetime = datetime(2023, 1, 1),
        end_date: datetime = datetime(2023, 12, 31)
    ) -> EventLog:
        """Generate a manufacturing process event log.
        
        Process flow: Order -> Plan -> Material Check -> [Produce|Wait] -> Quality Check -> Ship
        
        Args:
            num_cases: Number of process instances to generate
            start_date: Start date for event generation
            end_date: End date for event generation
            
        Returns:
            Generated EventLog
        """
        events = []
        time_range = (end_date - start_date).days
        
        for case_id in range(1, num_cases + 1):
            case_start = start_date + timedelta(days=random.randint(0, time_range))
            
            # Order
            events.append(Event(
                case_id=f"Order_{case_id:04d}",
                activity="Order",
                timestamp=case_start,
                resource="Customer",
                cost=0.0,
                duration=random.uniform(0.5, 1.0)
            ))
            
            # Plan
            plan_time = case_start + timedelta(hours=random.uniform(1, 12))
            events.append(Event(
                case_id=f"Order_{case_id:04d}",
                activity="Plan",
                timestamp=plan_time,
                resource=random.choice(["Planner_A", "Planner_B"]),
                cost=random.uniform(20, 60),
                duration=random.uniform(1.0, 3.0)
            ))
            
            # Material Check
            material_time = plan_time + timedelta(hours=random.uniform(2, 24))
            events.append(Event(
                case_id=f"Order_{case_id:04d}",
                activity="Material Check",
                timestamp=material_time,
                resource="Inventory_System",
                cost=random.uniform(5, 15),
                duration=random.uniform(0.2, 1.0)
            ))
            
            # Production decision - 85% can produce immediately
            production_time = material_time + timedelta(hours=random.uniform(1, 12))
            if random.random() < 0.85:
                # Produce
                events.append(Event(
                    case_id=f"Order_{case_id:04d}",
                    activity="Produce",
                    timestamp=production_time,
                    resource=random.choice(["Machine_A", "Machine_B", "Machine_C"]),
                    cost=random.uniform(100, 300),
                    duration=random.uniform(4.0, 12.0)
                ))
            else:
                # Wait for materials
                wait_time = production_time + timedelta(hours=random.uniform(24, 168))  # 1-7 days
                events.append(Event(
                    case_id=f"Order_{case_id:04d}",
                    activity="Wait",
                    timestamp=wait_time,
                    resource="System",
                    cost=random.uniform(10, 50),
                    duration=random.uniform(1.0, 7.0)
                ))
                
                # Produce after waiting
                production_time = wait_time + timedelta(hours=random.uniform(1, 6))
                events.append(Event(
                    case_id=f"Order_{case_id:04d}",
                    activity="Produce",
                    timestamp=production_time,
                    resource=random.choice(["Machine_A", "Machine_B", "Machine_C"]),
                    cost=random.uniform(100, 300),
                    duration=random.uniform(4.0, 12.0)
                ))
            
            # Quality Check
            quality_time = production_time + timedelta(hours=random.uniform(1, 6))
            events.append(Event(
                case_id=f"Order_{case_id:04d}",
                activity="Quality Check",
                timestamp=quality_time,
                resource=random.choice(["Inspector_A", "Inspector_B"]),
                cost=random.uniform(15, 45),
                duration=random.uniform(0.5, 2.0)
            ))
            
            # Ship
            ship_time = quality_time + timedelta(hours=random.uniform(2, 24))
            events.append(Event(
                case_id=f"Order_{case_id:04d}",
                activity="Ship",
                timestamp=ship_time,
                resource=random.choice(["Shipping_A", "Shipping_B"]),
                cost=random.uniform(25, 75),
                duration=random.uniform(1.0, 3.0)
            ))
        
        return EventLog(events)
