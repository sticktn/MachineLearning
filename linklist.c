//
// Created by guo on 22-9-12.
//
#include<malloc.h>

typedef struct intNode {
    int data;
    struct intNode *next;
} node;

// 构造函数
node *LinkList(int data) {
    node *head = malloc(sizeof(node));
    head->data = data;
    head->next = NULL;
}

void insertLast(node *head, int data) {
    node *n = head;
    while (n->next != NULL) {
        n = n->next;
    }
    node *newNode = malloc(sizeof(node));
    newNode->data = data;
    newNode->next = NULL;
    n->next = newNode;
}

void insertFirst(node *head, int data) {
    node *newNode = malloc(sizeof(node));
    newNode->data = data;
    newNode->next = head;
    head = newNode;
}

void insert(node *head, int index, int data) {
    int i;
    node *n = head;
    for (i = 0; i < index; i++) {
        n = n->next;
    }
    node *newNode = malloc(sizeof(node));
    newNode->data = data;
    newNode->next = n->next->next;
    n->next = newNode;
}

void delFirst(node *head) {
    node *n = head->next;
    free(head);
    head = n;
}

void delLast(node *head) {
    node *n = head;
    while (n->next->next != NULL) {
        n = n->next;
    }
    free(n->next);
    n->next = NULL;
}

void del(node *head, int index) {
    node *n = head;
    for (int i = 0; i < index - 1; ++i) {
        n = n->next;
    }
    node *dn = n->next;
    n->next = n->next->next;
    free(dn);
}

void changeData(node *head, int index, int data){
    node *n = head;
    for (int i = 0; i < index; ++i) {
        n = n->next;
    }
    n->data = data;
}

int getData(node *head, int index){
    node *n = head;
    for (int i = 0; i < index; ++i) {
        n = n->next;
    }
    return n->data;
}

void delAll(node *head){
    node *n = head;
    while (n != NULL){
        node *a = n->next;
        free(n);
        n = a;
    }
}

void toString(node *head){
    node *n = head;
    while (n != NULL){
        printf("%d, ",n->data);
        n = n->next;
    }
    printf("\n");
}

int main(){
    node *head = LinkList(3);
    for (int i = 1; i < 10; i++){
        insertLast(head,i);
    }
    toString(head);
    del(head,1);
    toString(head);
    changeData(head,1,188);
    toString(head);
    insert(head,1,285793);
    toString(head);
    delAll(head);
    return 0;
}