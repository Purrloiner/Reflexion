from reflection import make_reflection_CoT, make_reflection_react
from actor import CoT_actor, CoT_prompt, ReAct_actor, ReAct_prompt, prompt_GT, actor_GT, prompt_GT_Mem, actor_GT_Mem
import json
from evaluator import EM_score


with open('/home/sscc/Humane/data/hotpotqa_100_GT.json', 'r') as f:
    hotpotqa_100_GT = json.load(f)
    
with open('/home/sscc/Humane/data/hotpotqa_100_context.json', 'r') as f:
    hotpotqa_100_context = json.load(f)
    
with open('/home/sscc/Humane/data/hotpotqa_100.json', 'r') as f:
    hotpotqa_100 = json.load(f)
    

'''
# GT X =====================================================================================================

num = 100


#CoT only

def CoT():
    for i in range(num):
        index = i
        question = hotpotqa_100[i]["question"]

        for trial in range(3):
            print(f"Trial #{trial+1}")

            response = CoT_actor(index)
            prompt=CoT_prompt(index)
            
            print(prompt)
            print(response)

            score = EM_score(response, index)

            if score == 1:
                print("\nObservation : Answer is CORRECT")
                break

            print("\nObservation : Answer is INCORRECT")
            
            
# React only

def React():
    for i in range(num):   
        index = i
        question = hotpotqa_100[i]["question"]

        for trial in range(3):
            print(f"Trial #{trial+1}")

            response = ReAct_actor(index)
            prompt=ReAct_prompt(index)
            
            print(prompt)            
            print(response)

            score = EM_score(response, index)

            if score == 1:
                print("\nObservation : Answer is CORRECT")
                break

            print("\nObservation : Answer is INCORRECT")


# CoT + Reflexion

def CoT_Re():
    for i in range(num):        
        memory = []
        index = i
        question = hotpotqa_100[i]["question"]

        for trial in range(3):
            print(f"Trial #{trial+1}")

            response = CoT_actor(index, reflection=memory)
            prompt=CoT_prompt(index, reflection=memory)
            
            print(prompt)            
            print(response)

            score = EM_score(response, index)

            if score == 1:
                print("\nObservation : Answer is CORRECT")
                break

            print("\nObservation : Answer is INCORRECT")

            reflection = make_reflection_CoT(question, response)

            print("\nReflection :")
            print(reflection)

            experience = ( "\n\nReflection:\n" + reflection )

            memory.append(experience)

            if len(memory) > 3:
                memory.pop(0)


# React + Reflexion

def React_Re():
    for i in range(num):
        memory = []
        index = i
        question = hotpotqa_100[i]["question"]


        for trial in range(3):
            print(f"Trial #{trial+1}")

            response = ReAct_actor(index, reflection=memory)
            prompt=ReAct_prompt(index, reflection=memory)
            
            print(prompt)            
            print(response)

            score = EM_score(response, index)

            if score == 1:
                print("\nObservation : Answer is CORRECT")
                break

            print("\nObservation : Answer is INCORRECT")

            reflection = make_reflection_react(question, response)

            print("\nReflection :" )
            print(reflection)

            experience = ( "\n\nReflection:\n" + reflection )

            memory.append(experience)

            if len(memory) > 3:
                memory.pop(0)


# GT O ===========================================================================================================


# CoT only

def GT():
    for i in range(num):
        memory = []
        index = i
        question = hotpotqa_100[i]["question"]

        for trial in range(3):
            print(f"Trial #{trial+1}")

            response = actor_GT(index)
            prompt=prompt_GT(index)
            
            print(prompt)
            print(response)

            score = EM_score(response, index)

            if score == 1:
                print("\nObservation : Answer is CORRECT")
                break

            print("\nObservation : Answer is INCORRECT")


# CoT + Reflexion

def GT_Re():
    for i in range(num):       
        memory = []
        index = i
        question = hotpotqa_100[i]["question"]

        for trial in range(3):
            print(f"Trial #{trial+1}")

            response = actor_GT_Mem(index, memory)
            prompt=prompt_GT_Mem(index,memory)
            
            print(prompt)            
            print(response)

            score = EM_score(response, index)

            if score == 1:
                print("\nObservation : Answer is CORRECT")
                break

            print("\nObservation : Answer is INCORRECT")

            reflection = make_reflection_CoT(question, response)

            print("\nReflection :")
            print(reflection)

            experience = ( "\n\nReflection:\n" + reflection )

            memory.append(experience)

            if len(memory) > 3:
                memory.pop(0)



# CoT + EPM

def GT_Mem():
    for i in range(num):
        
        memory = []
        index = i
        question = hotpotqa_100[i]["question"]


        for trial in range(3):

            print(f"Trial #{trial+1}")

            response = actor_GT_Mem(index, memory)
            prompt=prompt_GT_Mem(index,memory)
            
            print(prompt)
            
            print(response)

            score = EM_score(response, index)

            if score == 1:
                print("\nObservation : Answer is CORRECT")
                break

            print("\nObservation : Answer is INCORRECT")

            experience = "Previous trajectory:\n" + response

            memory = [experience]


# CoT + EPM + Reflexion

def GT_Mem_Re():
    for i in range(num):       
        memory = []
        index = i
        question = hotpotqa_100[i]["question"]

        for trial in range(3):

            print(f"Trial #{trial+1}")

            response = actor_GT_Mem(index, memory)
            prompt=prompt_GT_Mem(index,memory)
            
            print(prompt)            
            print(response)

            score = EM_score(response, index)

            if score == 1:
                print("\nObservation : Answer is CORRECT")
                break

            print("\nObservation : Answer is INCORRECT")

            reflection = make_reflection_CoT(question, response)

            print("\nReflection :")
            print(reflection)

            experience = (
    "Previous trajectory:\n"
    + response
    + "\n\nReflection:\n"
    + reflection
)

            memory = [experience]

'''

def run_CoT(index):
    return CoT_actor(index)


def run_React(index):
    return ReAct_actor(index)


def run_CoT_Re(index, memory):
    return CoT_actor(index, reflection=memory)


def run_React_Re(index, memory):
    return ReAct_actor(index, reflection=memory)


def run_GT(index):
    return actor_GT(index)


def run_GT_Re(index, memory):
    return actor_GT_Mem(index, memory)


def run_GT_Mem(index, memory):
    return actor_GT_Mem(index, memory)


def run_GT_Mem_Re(index, memory):
    return actor_GT_Mem(index, memory)