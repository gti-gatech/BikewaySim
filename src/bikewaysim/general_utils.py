# General utility functions that can be use across the project

def print_elapsed_time(seconds):
    # Round the total seconds at the start
    seconds = round(seconds)
    
    # Calculate the elapsed days, hours, and minutes
    days = seconds // 86400  # 1 day = 86400 seconds
    hours = (seconds % 86400) // 3600  # 1 hour = 3600 seconds
    minutes = (seconds % 3600) // 60

    # Build the time string
    if days > 0:
        elapsed_time = f"{days:02} days {hours:02} hours {minutes:02} minutes"
    else:
        elapsed_time = f"{hours:02} hours {minutes:02} minutes"

    # Return the formatted elapsed time
    return elapsed_time