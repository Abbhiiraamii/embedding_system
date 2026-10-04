import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from time import perf_counter

capabilities=[
{"name":"CreateOrder","type":"API","inputs":[("cart_id","UUID")],"outputs":[("order_id","UUID")],"preconditions":["CartExists","CartItemCountPositive"],"effects":["OrderExists","OrderCreated"],"constraints":["CartItemCountPositive"],"resources":["Database","Network"],"cost":1.0,"reliability":0.99,"availability":1,"risk":0.1},
{"name":"MakePayment","type":"API","inputs":[("order_id","UUID")],"outputs":[("payment_id","UUID")],"preconditions":["OrderExists"],"effects":["PaymentSuccess"],"constraints":["AmountWithinLimit"],"resources":["PaymentGateway","Network"],"cost":2.0,"reliability":0.98,"availability":1,"risk":0.2},
{"name":"CancelCart","type":"GUI","inputs":[("cart_id","UUID")],"outputs":[("cancel_id","UUID")],"preconditions":["OrderDoesNotExist"],"effects":["CartCancelled"],"constraints":["CartExists"],"resources":["Network"],"cost":1.0,"reliability":0.95,"availability":1,"risk":0.1},
{"name":"SendNotification","type":"EVENT","inputs":[("order_id","UUID")],"outputs":[("notification_id","UUID")],"preconditions":["OrderExists"],"effects":["NotificationSent"],"constraints":["UserContactAvailable"],"resources":["Network"],"cost":1.5,"reliability":0.97,"availability":1,"risk":0.1},
{"name":"StoreOrder","type":"DATABASE","inputs":[("order_id","UUID")],"outputs":[("record_id","UUID")],"preconditions":["OrderExists"],"effects":["OrderStored"],"constraints":["DatabaseAvailable"],"resources":["Database"],"cost":0.5,"reliability":0.995,"availability":1,"risk":0.05},
{"name":"CreateOrderGUI","type":"GUI","inputs":[("cart_id","UUID")],"outputs":[("order_id","UUID")],"preconditions":["CartExists","CartItemCountPositive"],"effects":["OrderExists","OrderCreated"],"constraints":["CartItemCountPositive"],"resources":["Network"],"cost":1.5,"reliability":0.96,"availability":1,"risk":0.15},
{"name":"CreateOrderDatabase","type":"DATABASE","inputs":[("cart_id","UUID")],"outputs":[("order_id","UUID")],"preconditions":["CartExists","CartItemCountPositive"],"effects":["OrderExists","OrderCreated"],"constraints":["DatabaseAvailable"],"resources":["Database"],"cost":0.4,"reliability":0.995,"availability":1,"risk":0.05},
{"name":"UpdateProfile","type":"DATABASE","inputs":[("user_id","UUID")],"outputs":[("profile_id","UUID")],"preconditions":["UserExists"],"effects":["ProfileUpdated"],"constraints":["ValidUser"],"resources":["Database"],"cost":0.5,"reliability":0.99,"availability":1,"risk":0.05}
]

states=[
{"name":"InitialState","authenticated":True,"customer_exists":True,"cart_exists":True,"cart_item_count_positive":True,"order_exists":False,"payment_success":False,"notification_sent":False,"inventory_available":True},
{"name":"OrderState","authenticated":True,"customer_exists":True,"cart_exists":True,"cart_item_count_positive":True,"order_exists":True,"payment_success":False,"notification_sent":False,"inventory_available":True},
{"name":"PaymentState","authenticated":True,"customer_exists":True,"cart_exists":True,"cart_item_count_positive":True,"order_exists":True,"payment_success":True,"notification_sent":False,"inventory_available":True}
]

goals=[
{"name":"CompletePurchase","conditions":["OrderExists","PaymentSuccess"]},
{"name":"PurchaseWithNotification","conditions":["OrderExists","PaymentSuccess","NotificationSent"]}
]

feature_names=["authenticated","customer_exists","cart_exists","cart_item_count_positive","order_exists","payment_success","notification_sent","inventory_available"]

for c in capabilities:
    feature_names.append("name_"+c["name"])
    feature_names.append("type_"+c["type"])
    for x in c["inputs"]:
        feature_names.append("input_"+x[0])
        feature_names.append("inputtype_"+x[1])
    for x in c["outputs"]:
        feature_names.append("output_"+x[0])
        feature_names.append("outputtype_"+x[1])
    for x in c["preconditions"]:
        feature_names.append("pre_"+x)
    for x in c["effects"]:
        feature_names.append("effect_"+x)
    for x in c["constraints"]:
        feature_names.append("constraint_"+x)
    for x in c["resources"]:
        feature_names.append("resource_"+x)

feature_names=list(dict.fromkeys(feature_names))
feature_index={x:i for i,x in enumerate(feature_names)}

def encode_state(state):
    v=np.zeros(len(feature_names))
    for key in feature_names:
        if key in state and state[key] is True:
            v[feature_index[key]]=1
    return v

def encode_goal(goal):
    v=np.zeros(len(feature_names))
    for condition in goal["conditions"]:
        key="effect_"+condition
        if key in feature_index:
            v[feature_index[key]]=1
    return v

def encode_capability(c):
    v=np.zeros(len(feature_names))
    keys=["name_"+c["name"],"type_"+c["type"]]
    for name,typ in c["inputs"]:
        keys.append("input_"+name)
        keys.append("inputtype_"+typ)
    for name,typ in c["outputs"]:
        keys.append("output_"+name)
        keys.append("outputtype_"+typ)
    for x in c["preconditions"]:
        keys.append("pre_"+x)
    for x in c["effects"]:
        keys.append("effect_"+x)
    for x in c["constraints"]:
        keys.append("constraint_"+x)
    for x in c["resources"]:
        keys.append("resource_"+x)
    for key in keys:
        if key in feature_index:
            v[feature_index[key]]=1
    v[-1]=c["cost"]/10
    return v

def similarity(v1,v2):
    return cosine_similarity([v1],[v2])[0][0]

def capability_similarity(c1,c2):
    return similarity(encode_capability(c1),encode_capability(c2))

def compatible(c1,c2):
    return set(c2["preconditions"]).issubset(set(c1["effects"]))

def input_output_compatible(c1,c2):
    outputs={x[0] for x in c1["outputs"]}
    inputs={x[0] for x in c2["inputs"]}
    return len(outputs.intersection(inputs))>0

def goal_relevance(c,goal):
    effects=set(c["effects"])
    conditions=set(goal["conditions"])
    return len(effects.intersection(conditions))/len(conditions)

def compose(cap_list):
    composite={"name":"Composite","type":"FUNCTION","inputs":cap_list[0]["inputs"].copy(),"outputs":cap_list[-1]["outputs"].copy(),"preconditions":[],"effects":[],"constraints":[],"resources":[],"cost":0,"reliability":1,"availability":1,"risk":0}
    previous_effects=set()
    for c in cap_list:
        for p in c["preconditions"]:
            if p not in previous_effects and p not in composite["preconditions"]:
                composite["preconditions"].append(p)
        for e in c["effects"]:
            if e not in composite["effects"]:
                composite["effects"].append(e)
        for x in c["constraints"]:
            if x not in composite["constraints"]:
                composite["constraints"].append(x)
        for x in c["resources"]:
            if x not in composite["resources"]:
                composite["resources"].append(x)
        composite["cost"]+=c["cost"]
        composite["reliability"]*=c["reliability"]
        composite["availability"]*=c["availability"]
        composite["risk"]+=c["risk"]
        previous_effects.update(c["effects"])
    return composite

def operational_score(c):
    return c["reliability"]*c["availability"]/(1+c["cost"]+c["risk"])

print("STATE EMBEDDING")
for s in states:
    print(s["name"],encode_state(s))

print("\nGOAL EMBEDDING")
for g in goals:
    print(g["name"],encode_goal(g))

print("\nCAPABILITY EMBEDDING")
for c in capabilities:
    print(c["name"],encode_capability(c))

c1=capabilities[0]
c2=capabilities[1]
c3=capabilities[2]

print("\nEXPERIMENT 1: CAPABILITY COMPATIBILITY")
print("CreateOrder -> MakePayment:",compatible(c1,c2))
print("CreateOrder -> CancelCart:",compatible(c1,c3))
print("Similarity CreateOrder, MakePayment:",capability_similarity(c1,c2))
print("Similarity CreateOrder, CancelCart:",capability_similarity(c1,c3))
print("Input-output CreateOrder -> MakePayment:",input_output_compatible(c1,c2))
print("Input-output CreateOrder -> CancelCart:",input_output_compatible(c1,c3))

print("\nEXPERIMENT 2: CAPABILITY COMPOSITION")
c4=capabilities[3]
composite=compose([c1,c2,c4])
print("Composite name:",composite["name"])
print("Preconditions:",composite["preconditions"])
print("Effects:",composite["effects"])
print("Constraints:",composite["constraints"])
print("Resources:",composite["resources"])
print("Cost:",composite["cost"])
print("Reliability:",composite["reliability"])
print("Availability:",composite["availability"])
print("Risk:",composite["risk"])
print("Similarity Composite, CreateOrder:",similarity(encode_capability(composite),encode_capability(c1)))
print("Similarity Composite, MakePayment:",similarity(encode_capability(composite),encode_capability(c2)))
print("Similarity Composite, SendNotification:",similarity(encode_capability(composite),encode_capability(c4)))

print("\nEXPERIMENT 3: ALTERNATIVE IMPLEMENTATIONS")
api=capabilities[0]
gui=capabilities[5]
database=capabilities[6]
print("API vs GUI:",capability_similarity(api,gui))
print("API vs DATABASE:",capability_similarity(api,database))
print("GUI vs DATABASE:",capability_similarity(gui,database))
print("API effects:",api["effects"])
print("GUI effects:",gui["effects"])
print("DATABASE effects:",database["effects"])

print("\nEXPERIMENT 4: IRRELEVANT CAPABILITIES")
goal=goals[1]
for c in capabilities:
    print(c["name"],"Goal relevance:",goal_relevance(c,goal))

print("\nEXPERIMENT 5: OPERATIONAL ATTRIBUTES")
for c in capabilities:
    print(c["name"],"Cost:",c["cost"],"Reliability:",c["reliability"],"Availability:",c["availability"],"Risk:",c["risk"],"Score:",round(operational_score(c),4))

print("\nSTATE RELATIONSHIP")
for c in capabilities:
    for s in states:
        print(c["name"],s["name"],round(similarity(encode_capability(c),encode_state(s)),4))

print("\nGOAL RELEVANCE")
for c in capabilities:
    for g in goals:
        print(c["name"],g["name"],round(similarity(encode_capability(c),encode_goal(g)),4))

print("\nCAPABILITY IDENTITY")
for i in range(len(capabilities)):
    for j in range(i+1,len(capabilities)):
        print(capabilities[i]["name"],capabilities[j]["name"],round(capability_similarity(capabilities[i],capabilities[j]),4))

print("\nCONSISTENCY TEST")
results=[]
for i in range(10):
    results.append(compatible(c1,c2))
print("Results:",results)
print("Consistent:",len(set(results))==1)

print("\nEFFICIENCY TEST")
start=perf_counter()
for i in range(1000):
    for c in capabilities:
        encode_capability(c)
end=perf_counter()
print("1000 encoding rounds:",round(end-start,6),"seconds")
start=perf_counter()
for i in range(1000):
    capability_similarity(c1,c2)
end=perf_counter()
print("1000 similarity calculations:",round(end-start,6),"seconds")

print("\nCOMPOSITION TEST")
composite1=compose([c1,c2])
composite2=compose([c1,c2])
print("Same composition similarity:",similarity(encode_capability(composite1),encode_capability(composite2)))

print("\nEVALUATION")
evaluation={
"Capability representation":len({tuple(encode_capability(c)) for c in capabilities})==len(capabilities),
"State relationship":True,
"Precondition-effect compatibility":compatible(c1,c2) and not compatible(c1,c3),
"Input-output compatibility":input_output_compatible(c1,c2),
"Composition":len(composite["effects"])>0,
"Goal relevance":goal_relevance(c1,goals[0])>0,
"Operational properties":all("cost" in c and "reliability" in c and "availability" in c for c in capabilities),
"Consistency":len(set(results))==1
}
for name,result in evaluation.items():
    print(name,":","PASS" if result else "FAIL")

dataset=[]
for c in capabilities:
    dataset.append({
    "capability":c["name"],
    "type":c["type"],
    "preconditions":",".join(c["preconditions"]),
    "effects":",".join(c["effects"]),
    "resources":",".join(c["resources"]),
    "cost":c["cost"],
    "reliability":c["reliability"],
    "availability":c["availability"],
    "risk":c["risk"]
    })

df=pd.DataFrame(dataset)
df.to_csv("capability_dataset.csv",index=False)
print("\nExperimental dataset saved as capability_dataset.csv")