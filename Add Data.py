from datetime import datetime
todayDate = datetime.now()
current_year = todayDate.strftime('%Y')
current_time = todayDate.strftime("%H:%M:%S")
print("Current time:", current_time)
if current_year > '2023':
    print("Current Year True:")
else:
    print("Current  Year false:")