from rag import ask_rag

questions = [

    "Tell me about the Leave Policy.",

    "How long is it?",

    "What about maternity leave?",

    "And paternity?",

    "Does it apply during probation?"

]

for question in questions:

    print("=" * 80)

    print("USER:")
    print(question)

    answer, context = ask_rag(question)

    print("\nASSISTANT:")
    print(answer)

    print()