from reflection import make_reflection_CoT, make_reflection_react

def make_experience(response, reflection):

    experience = ""

    experience += "Previous Trajectory:\n"
    experience += response

    experience += "\n\nReflection:\n"
    experience += reflection

    return experience