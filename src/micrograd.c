#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Definition of the Value struct with added backpropagation support
typedef struct Value {
    double data;
    double grad;
    struct Value** children;
    size_t child_count;
    char op[10];
    void (*_backward)(struct Value* self);
    int requires_grad;
} Value;

// Forward declarations of backward functions
void Value_add_backward(Value* result);
void Value_mul_backward(Value* result);
void Value_relu_backward(Value* result);

// Function prototypes for backpropagation
void Value_backward(Value* root);
void Value_zero_grad(Value* v);
void Value_free(Value* v);

// Utility functions for Value with backpropagation support
Value* Value_create(double data, int requires_grad) {
    Value* v = (Value*)malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->children = NULL;
    v->child_count = 0;
    v->_backward = NULL;
    v->requires_grad = requires_grad;
    strcpy(v->op, "");
    return v;
}

// Add child tracking for computational graph
void Value_add_child(Value* parent, Value* child) {
    parent->children = realloc(parent->children, 
        (parent->child_count + 1) * sizeof(Value*));
    parent->children[parent->child_count] = child;
    parent->child_count++;
}

// Enhanced addition with backpropagation support
void Value_add(Value* result, Value* a, Value* b) {
    result->data = a->data + b->data;
    result->requires_grad = a->requires_grad || b->requires_grad;
    strcpy(result->op, "+");
    
    if (result->requires_grad) {
        Value_add_child(result, a);
        Value_add_child(result, b);
        
        result->_backward = Value_add_backward;
    }
}

// Backward pass for addition
void Value_add_backward(Value* result) {
    if (result->children[0]->requires_grad) {
        result->children[0]->grad += result->grad;
    }
    if (result->children[1]->requires_grad) {
        result->children[1]->grad += result->grad;
    }
}

// Enhanced multiplication with backpropagation support
void Value_mul(Value* result, Value* a, Value* b) {
    result->data = a->data * b->data;
    result->requires_grad = a->requires_grad || b->requires_grad;
    strcpy(result->op, "*");
    
    if (result->requires_grad) {
        Value_add_child(result, a);
        Value_add_child(result, b);
        
        result->_backward = Value_mul_backward;
    }
}

// Backward pass for multiplication
void Value_mul_backward(Value* result) {
    if (result->children[0]->requires_grad) {
        result->children[0]->grad += result->children[1]->data * result->grad;
    }
    if (result->children[1]->requires_grad) {
        result->children[1]->grad += result->children[0]->data * result->grad;
    }
}

// ReLU with backpropagation
void Value_relu(Value* result, Value* a) {
    result->data = fmax(0.0, a->data);
    result->requires_grad = a->requires_grad;
    strcpy(result->op, "ReLU");
    
    if (result->requires_grad) {
        Value_add_child(result, a);
        result->_backward = Value_relu_backward;
    }
}

// Backward pass for ReLU
void Value_relu_backward(Value* result) {
    if (result->children[0]->requires_grad) {
        result->children[0]->grad += (result->data > 0) ? result->grad : 0;
    }
}

// Recursive backpropagation
void Value_backward(Value* root) {
    // Accumulate gradients through the graph
    if (root->_backward) {
        root->_backward(root);
    }
    
    // Recursively call backward on children
    for (int i = root->child_count - 1; i >= 0; i--) {
        Value_backward(root->children[i]);
    }
}

// Zero out gradients before a new backward pass
void Value_zero_grad(Value* v) {
    v->grad = 0.0;
    for (size_t i = 0; i < v->child_count; i++) {
        Value_zero_grad(v->children[i]);
    }
}

// Free memory for a Value and its children
void Value_free(Value* v) {
    if (v == NULL) return;
    
    // Recursively free children
    for (size_t i = 0; i < v->child_count; i++) {
        Value_free(v->children[i]);
    }
    
    // Free children array if it exists
    if (v->children) {
        free(v->children);
    }
    
    // Free the Value itself
    free(v);
}

// Print the computational graph with indentation
void print_computational_graph(Value* v, int depth) {
    // Indent based on depth
    for (int i = 0; i < depth; i++) {
        printf("  ");
    }
    
    // Print current node details
    printf("Value: %.2f, Grad: %.2f, Op: %s, Requires Grad: %s\n", 
           v->data, v->grad, 
           strlen(v->op) > 0 ? v->op : "None", 
           v->requires_grad ? "Yes" : "No");
    
    // Recursively print children
    for (size_t i = 0; i < v->child_count; i++) {
        print_computational_graph(v->children[i], depth + 1);
    }
}

// Main function with interactive input
int main() {
    double input1, input2, bias_val;
    int computation_type;
    
    // Welcome message
    printf("Welcome to Micrograd C - An Autograd Demonstration\n");
    
    // Get user inputs
    printf("Enter first input value: ");
    scanf("%lf", &input1);
    
    printf("Enter second input value: ");
    scanf("%lf", &input2);
    
    printf("Enter bias value: ");
    scanf("%lf", &bias_val);
    
    // Computation type selection
    printf("\nSelect computation type:\n");
    printf("1. Multiplication + Addition + ReLU\n");
    printf("2. Only Multiplication\n");
    printf("3. Only Addition\n");
    printf("Enter your choice (1-3): ");
    scanf("%d", &computation_type);
    
    // Create input values with gradient tracking
    Value* a = Value_create(input1, 1);  // requires gradient
    Value* b = Value_create(input2, 1);  // requires gradient
    Value* bias = Value_create(bias_val, 1); // requires gradient
    
    // Intermediate and final results
    Value* result1 = Value_create(0, 1);
    Value* result2 = Value_create(0, 1);
    Value* final_output = Value_create(0, 1);
    
    // Perform selected computation
    switch(computation_type) {
        case 1:
            // Multiply, add bias, then apply ReLU
            Value_mul(result1, a, b);
            Value_add(result2, result1, bias);
            Value_relu(final_output, result2);
            break;
        case 2:
            // Only multiplication
            Value_mul(final_output, a, b);
            break;
        case 3:
            // Only addition
            Value_add(final_output, a, bias);
            break;
        default:
            printf("Invalid computation type. Defaulting to Multiplication.\n");
            Value_mul(final_output, a, b);
    }
    
    // Zero gradients before backward pass
    Value_zero_grad(final_output);
    
    // Set the gradient of the final node to 1.0
    final_output->grad = 1.0;
    
    // Perform backpropagation
    Value_backward(final_output);
    
    // Print computational results
    printf("\nComputational Results\n");
    printf("--------------------\n");
    printf("Output: %.2f\n", final_output->data);
    
    // Print gradients
    printf("\nGradients\n");
    printf("---------\n");
    printf("Input1 gradient: %.2f\n", a->grad);
    printf("Input2 gradient: %.2f\n", b->grad);
    printf("Bias gradient: %.2f\n", bias->grad);
    
    // Print computational graph
    printf("\nComputational Graph\n");
    printf("-------------------\n");
    print_computational_graph(final_output, 0);
    
    // Free memory
    Value_free(final_output);
    
    return 0;
}
