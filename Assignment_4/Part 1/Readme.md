## **Part 1**

### **Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github:-**
* Use exactly the same values for all variables as used in the class
* Take a screenshot, and show that screenshot in the readme file
* Excel file must be there for us to cross-check the image shown on readme (no image = no score)
* Explain each major step
* Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 
* Upload all this to GitHub and then write all above as part 1 of your README.md file. 
* Submit details to S4 - Assignment QnA. 

#### **Flow Diagram:-**
![Flow diagram](https://user-images.githubusercontent.com/50147394/120015542-b8922900-bfe3-11eb-8937-d6d8805424fd.png)


#### NETWORK ARCHITECTURE:
* There are 3 layers in the network 
ie: Input layer, hidden layer and output layer.

#### NAMING CONVENTION OF THE NETWORK PARAMETERS: 
* i1 and i2 are the inputs from the input layer.
* w1,w2....w8 are the randomly initialized weights used in the network.
* h1 and h2 are the weighted inputs.
* a_h1 and a_h2 are the weighted inputs after activation function is applied.
* o1 and o2 are the output layer values.
* a_o1 and a_o2 are the outputs after activation is used.
* t1 and t2 are the target values.
* E1 and E2 are the losses for the outputs.
* E_Total is the total loss.


 #### FORWARD PASS OF THE NETWORK
        # Weighted input obtained by multiplying the random weights
	h1 = w1*i1+w2*i2		
	h2 = w3*i1+w4*i2
	
	# These are the activated weighted inputs
	a_h1 = σ(h1) = 1/(1+exp(-h1))		
	a_h2 = σ(h2)		
	
	#The activated weighted inputs from the previous layer are again transformed linearly 
	o1 = w5*a_h1+w6*a_h2		
	o2 = w7*a_h1+w8*a_h2		
	
	# Activating the newly generated weighted inputs from the previous layer
	a_o1 = σ(o1)		
	a_o2 = σ(o2)		
	
	# Calculating the total loss is sum of loss of individual classifications
	E_total = E1 + E2	
	
	# calculating the loss of classifying as t1
	E1 = 1/2 * (t1 - a_o1)2	
	
	# Calculating the loss of classifying as t2
	E2 = 1/2 * (t2 - a_o2)2		


 #### BACKWARD PASS OF THE NETWORK
        #Calculation of gradient of loss w.r.t W5
	∂E_total/∂w5 = ∂(E1 + E2)∂w5							
	∂E_total/∂w5 = ∂E1/∂w5							
	∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5
	
	∂E1/∂a_o1 = ∂(1/2*(t1 - a1_o1)2)/∂a_o1 = (a_o1 - t1)							
	∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = a_o1*(1 - a_o1)							
	∂o1/∂w5 = a_h1							

	∂E_total/∂w5 = (a_o1 - t1)*a_o1*(1 - a_o1)*a_h1	
	
	#Similarly these are the gradients of loss w.r.t w6, w7, w8
	∂E_total/∂w6 = (a_o1 - t1)*a_o1*(1 - a_o1)*a_h2							
	∂E_total/∂w7 = (a_o2 - t2)*a_o2*(1 - a_o2)*a_h1							
	∂E_total/∂w8 = (a_o2 - t2)*a_o2*(1 - a_o2)*a_h2							

	∂E1/∂a_h1 = (a_o1 - t1)*a_o1*(1 - a_o1)*w5							
	∂E2/∂a_h1 = (a_o2 - t2)*a_o2*(1 - a_o2)*w7
	
	∂E_total/∂a_h1 = (a_o1 - t1)*a_o1*(1 - a_o1)*w5 + (a_o2 - t2)*a_o2*(1 - a_o2)*w7				
	∂E_total/∂a_h2 = (a_o1 - t1)*a_o1*(1 - a_o1)*w6 + (a_o2 - t2)*a_o2*(1 - a_o2)*w8		

       #Similarly these are the gradients of loss w.r.t w1, w2, w3, w4
	∂E_total/∂w1 = ∂E_total/∂a_h1*∂a_h1/∂h1*∂h1/∂w1									
	∂E_total/∂w2 = ∂E_total/∂a_h1*∂a_h1/∂h1*∂h1/∂w2									
	∂E_total/∂w3 = ∂E_total/∂a_h2*∂a_h2/∂h2*∂h2/∂w3		
	∂E_total/∂w4 = ∂E_total/∂a_h2*∂a_h2/∂h2*∂h2/∂w4	

	∂E_total/∂w1 = ((a_o1 - t1)*a_o1*(1 - a_o1)*w5 + (a_o2 - t2)*a_o2*(1 - a_o2)*w7)*a_h1*(1 - a_h1)*i1									
	∂E_total/∂w2 = ((a_o1 - t1)*a_o1*(1 - a_o1)*w5 + (a_o2 - t2)*a_o2*(1 - a_o2)*w7)*a_h1*(1 - a_h1)*i2									
	∂E_total/∂w3 = ((a_o1 - t1)*a_o1*(1 - a_o1)*w6 + (a_o2 - t2)*a_o2*(1 - a_o2)*w8)*a_h2*(1 - a_h2)*i1									
	∂E_total/∂w4 = ((a_o1 - t1)*a_o1*(1 - a_o1)*w6 + (a_o2 - t2)*a_o2*(1 - a_o2)*w8)*a_h2*(1 - a_h2)*i2									

#### Screenshot of the Excel file uploaded

![Excel work](https://user-images.githubusercontent.com/50147394/120016271-af558c00-bfe4-11eb-8c11-4b6d00327d1a.png)


#### Total Loss Graph for various learning rates(η)

![Total Loss Graph](https://user-images.githubusercontent.com/50147394/120014884-e0cd5800-bfe2-11eb-93e8-95a99d86c4eb.png)



## **Contributors:-**

1. Avinash Ravi 
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta

