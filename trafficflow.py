import numpy as np

def flow_to_speed(self, flow):
    if flow <= 1500:
        a = -1.4648375
        b = 93.75
        c = -flow
        
        #quadratic curve formula
        speed = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        
    else:
        speed = 32 - (flow - 1500) / 100  
    
    #speed limit according to document
    speed = min(speed, 60)
    
    #prevent numbers from being unrealistic
    speed = max(speed, 5)
    
    return speed

def travel_time(self, distance, predicted_flow):

    speeds = self.flow_to_speed(predicted_flow)
    avg_speed = np.mean(speeds)

    time = distance / avg_speed

    seconds = time * 3600 + 30

    return seconds, avg_speed