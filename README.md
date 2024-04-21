# mirAI

## Overview
mirAI is a full-stack web application that classifies brain tumors using a model that was trained and fine-tuned using Intel Developer Cloud.

## Implementation
For the front-end we used one of the sponsors' technology, Reflex, which communicates with a Flask back-end through HTTP requests. The backend then interacts with the ssh Intel Developer Cloud instance of where the model that we trained is hosted. 

## Getting Started
- Run `reflex` to start the front-end
```
run reflex
init reflex
```
