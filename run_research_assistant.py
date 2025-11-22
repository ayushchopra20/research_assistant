import research_assistant as RA
if __name__ == "__main__":

    query = "What is Edge Computing?"
    print(f"\nGeneral Query: {query}\n")
    answer = RA.answer_question(query)
    print("Answer:\n")
    print(answer)


    query = "According to the file that you have just read, why it is important now a days?"
    print(f"\nFollow-up Query: {query}\n")
    answer = RA.answer_question(query)
    print("Answer:\n")
    print(answer)

    query = "Give me a summary of the contents"
    print(f"\nSummary Query: {query}\n")
    answer = RA.answer_question(query)
    print("Answer:\n")
    print(answer)