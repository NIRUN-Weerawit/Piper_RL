# check_api.py
import sys
from isaacgym import gymapi

print("-" * 50)
print(f"Using Python executable: {sys.executable}")
print(f"Location of gymapi module: {gymapi.__file__}")
print("-" * 50)

# Check if the attribute exists before trying to use it
if hasattr(gymapi, 'GymLightProperties'):
    print("✅ SUCCESS: 'GymLightProperties' was found in gymapi.")
    try:
        props = gymapi.GymLightProperties()
        print("✅ SUCCESS: Instantiated gymapi.GymLightProperties().")
    except Exception as e:
        print(f"❌ FAILED: Could not instantiate object. Error: {e}")
else:
    print("❌ FAILED: 'GymLightProperties' was NOT found in gymapi.")
    print("\nThis suggests an issue with your Isaac Gym version or installation.")
    print("Available attributes in gymapi (first 20):")
    # Print some of the available attributes to see what is there
    print(dir(gymapi)[:20])

print("-" * 50)
