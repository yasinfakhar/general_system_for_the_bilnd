from config import Config
def get_age_cathegory(age):
    
    if age <= Config.baby_age : return "baby"
    
    elif age <= Config.child_age : return "child"
    
    elif age <= Config.teenager_age : return "teenager"
    
    elif age <= Config.young_age : return "young"
    
    elif age <= Config.middle_age : return "middle_aged"
    
    elif age <= Config.adult_age : return "adult"
    
    elif age <= Config.old_age : return "old"
    