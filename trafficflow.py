import numpy as np

def flow_to_speed(self, flow):
    # Convert traffic flow to speed using a quadratic curve for flow <= 1500
    if flow <= 1500:
        a = -1.4648375
        b = 93.75
        c = -flow
        
        # Use the quadratic formula to solve for speed
        speed = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        
    else:
        # For higher flows, use a linear decrease
        speed = 32 - (flow - 1500) / 100  
    
    # Enforce speed limit (max 60 km/h)
    speed = min(speed, 60)
    
    # Prevent unrealistic speeds (min 5 km/h)
    speed = max(speed, 5)
    
    return speed

def travel_time(self, distance, predicted_flow):
    # Calculate travel time given distance and predicted flow
    speeds = self.flow_to_speed(predicted_flow)

    # Average speed (handles array input)
    avg_speed = np.mean(speeds)
    
    # Time in hours
    time = distance / avg_speed
    
    # Convert to seconds and add 30s intersection delay
    seconds = time * 3600 + 30
    
    return time, seconds, avg_speed