"""
backend/agents/rag_agent/evals/eval_dataset.py
================================================
Golden Q&A evaluation dataset for RAGAS evaluation.

31 questions — one per HR document — plus a few cross-document multi-hop questions.
Ground truth answers are concise factual summaries drawn from each document's
most policy-critical section.

These are used by eval_runner.py to compute Faithfulness, AnswerRelevancy,
ContextRecall, and ContextPrecision via the RAGAS framework.
"""

from __future__ import annotations

from typing import TypedDict


class EvalSample(TypedDict):
    question: str
    ground_truth: str
    source_document: str


GOLDEN_DATASET: list[EvalSample] = [
    {
        "question": "What is the company's alcohol policy regarding employees under the influence at work?",
        "ground_truth": "Employees must not be under the influence of alcohol while at work. The company may require alcohol testing and disciplinary action including dismissal may follow if an employee is found to be intoxicated.",
        "source_document": "Alcohol-Policy.pdf",
    },
    {
        "question": "What are the main prohibitions under the anti-bribery policy?",
        "ground_truth": "Employees must not offer, give, request, or accept bribes. This applies to dealings with public officials and private individuals. Facilitation payments are also prohibited.",
        "source_document": "Anti-Bribery-Policy.docx",
    },
    {
        "question": "What restrictive covenants are typically included in the employment contract offer letter?",
        "ground_truth": "The employment contract offer letter includes restrictions such as non-competition, non-solicitation of clients and employees, and confidentiality obligations that survive termination of employment.",
        "source_document": "Basic-Employment-Contract-With-Restrictive-Covenants-Offer-Letter.docx",
    },
    {
        "question": "What are the key requirements under the BYOD policy for personal devices used for work?",
        "ground_truth": "Employees must register personal devices, install required security software, use encryption, and ensure company data is not mixed with personal data. The company retains the right to remotely wipe work data from personal devices.",
        "source_document": "Bring-Your-Own-Device-BYOD-Policy.pdf",
    },
    {
        "question": "What are employees allowed to use company email and internet for?",
        "ground_truth": "Company email and internet are for business use. Limited personal use may be permitted but must not include accessing inappropriate content, excessive social media use during work hours, or sending confidential information to personal accounts.",
        "source_document": "Communications-Email-Internet-and-Social-Media-Policy.pdf",
    },
    {
        "question": "What happens at the end of a successful probationary period?",
        "ground_truth": "At the end of a successful probationary period, the employee receives written confirmation of their employment status, confirming they have passed probation and are now employed on permanent terms.",
        "source_document": "Confirmation-Of-Employment-And-End-Of-Probationary-Period-Letter.docx",
    },
    {
        "question": "What are the grounds and process for dismissal at the end of a probationary period?",
        "ground_truth": "If performance or conduct during probation is unsatisfactory, the employer may dismiss the employee at the end of the probationary period with appropriate notice, as set out in the dismissal letter.",
        "source_document": "Dismissal-at-End-of-Probationary-Period-Letter.docx",
    },
    {
        "question": "What is the company's policy on drug testing and what are the consequences of a positive test?",
        "ground_truth": "The company may require drug testing. A positive test result may lead to disciplinary action including dismissal. Employees who are dependent on drugs may be referred to occupational health.",
        "source_document": "Drugs-Policy.pdf",
    },
    {
        "question": "What types of expenses can employees claim, and what is the approval process?",
        "ground_truth": "Employees can claim reasonable business expenses such as travel, accommodation, and meals incurred in the course of work. Expenses must be pre-approved where required and submitted with receipts within a set timeframe.",
        "source_document": "Employee-Expenses-Policy.docx",
    },
    {
        "question": "What protected characteristics does the equal opportunities policy cover?",
        "ground_truth": "The equal opportunities policy covers all protected characteristics under the Equality Act 2010, including age, disability, gender reassignment, marriage and civil partnership, pregnancy and maternity, race, religion or belief, sex, and sexual orientation.",
        "source_document": "Equal-Opportunity-and-Diversity-Policy.docx",
    },
    {
        "question": "What equipment is listed on the equipment receipt form and what are the employee's obligations?",
        "ground_truth": "The equipment receipt form lists company equipment issued to the employee (e.g., laptop, phone). The employee acknowledges receipt and agrees to return equipment in good condition upon termination.",
        "source_document": "Equipment-Receipt-Form.docx",
    },
    {
        "question": "What topics are covered in a typical exit interview?",
        "ground_truth": "Exit interviews typically cover reasons for leaving, views on the role and management, suggestions for improvement, and any concerns the employee wishes to raise before departure.",
        "source_document": "Exit-Interview.docx",
    },
    {
        "question": "Under what circumstances can a probationary period be extended and for how long?",
        "ground_truth": "A probationary period may be extended if the employee's performance or conduct has not met the required standard. The extension period and review criteria are communicated in writing to the employee.",
        "source_document": "Extension-of-Probationary-Period-Letter.docx",
    },
    {
        "question": "What are the key terms that distinguish a fixed-term employment contract from a permanent one?",
        "ground_truth": "A fixed-term contract specifies a defined end date or completion of a specific task. The employee is entitled to the same rights as permanent employees and must receive written notice if the contract will not be renewed.",
        "source_document": "Fixed-Term-Employment-Contract.docx",
    },
    {
        "question": "How does the harassment and bullying policy define bullying in the workplace?",
        "ground_truth": "Bullying is defined as offensive, intimidating, malicious, or insulting behaviour involving an abuse of power that makes the recipient feel upset, threatened, humiliated, or vulnerable.",
        "source_document": "Harassment-and-bullying-policy-3.docx",
    },
    {
        "question": "What does the induction policy require to be completed in the first week of employment?",
        "ground_truth": "The induction policy requires new starters to complete mandatory activities in the first week, including health and safety training, IT setup, HR paperwork, and a tour of facilities, as outlined in the induction checklist.",
        "source_document": "Induction-Policy-1.docx",
    },
    {
        "question": "What information must be included in a letter acknowledging an employee's resignation?",
        "ground_truth": "The resignation acknowledgement letter must confirm the last working day, the notice period, any garden leave arrangements, and instructions for returning company property.",
        "source_document": "Letter-acknowledging-resignation.docx",
    },
    {
        "question": "What details must a promotion confirmation letter include?",
        "ground_truth": "A promotion confirmation letter must include the new job title, new salary, effective date of promotion, any changes to terms and conditions, and confirmation that existing terms remain unchanged unless otherwise stated.",
        "source_document": "Letter-confirming-promotion.docx",
    },
    {
        "question": "What are the conditions under which garden leave can be imposed?",
        "ground_truth": "Garden leave can be imposed during the notice period where the employer requires the employee to stay away from the workplace, not contact clients or colleagues, and remain available to the employer while still receiving full pay.",
        "source_document": "Letter-putting-employee-on-garden-leave.docx",
    },
    {
        "question": "What is the total statutory maternity leave entitlement and how is pay structured?",
        "ground_truth": "Employees are entitled to up to 52 weeks of maternity leave (26 weeks ordinary + 26 weeks additional). Statutory Maternity Pay (SMP) is paid for up to 39 weeks: 90% of average weekly earnings for the first 6 weeks, then a standard rate for the remaining 33 weeks.",
        "source_document": "Maternity-Policy.docx",
    },
    {
        "question": "What information does the new starter form collect from employees?",
        "ground_truth": "The new starter form collects personal details, emergency contact information, bank details for payroll, tax information (P45 or starter declaration), and right-to-work documentation.",
        "source_document": "New-Starter-Form-1.docx",
    },
    {
        "question": "What are the notice period requirements for different lengths of service?",
        "ground_truth": "Notice periods increase with length of service. Statutory minimums are: 1 week for less than 2 years, then one week per year of service up to a maximum of 12 weeks. Contractual notice may be longer.",
        "source_document": "Notice-Periods-Policy.pdf",
    },
    {
        "question": "What is the statutory paternity leave entitlement for birth?",
        "ground_truth": "Eligible employees are entitled to 1 or 2 consecutive weeks of statutory paternity leave (SPL) following the birth of a child. Statutory Paternity Pay (SPP) is paid at the lower of the standard rate or 90% of average weekly earnings.",
        "source_document": "Paternity-Leave-Policy-Birth.docx",
    },
    {
        "question": "What is the standard duration of a probationary period and what review process applies?",
        "ground_truth": "The standard probationary period is typically 3 to 6 months. During this time, regular review meetings are held to assess performance. The period may be extended or ended with notice if performance is unsatisfactory.",
        "source_document": "Probationary-Periods-Policy.docx",
    },
    {
        "question": "What criteria determine whether an employee is at risk of redundancy?",
        "ground_truth": "Employees may be at risk of redundancy when there is a reduction in the need for employees to do work of a particular kind. Selection criteria may include skills, performance, attendance, and business need, applied fairly and consistently.",
        "source_document": "Redundancy-Policy (1).docx",
    },
    {
        "question": "At what age can employees retire and what support does the company provide?",
        "ground_truth": "There is no mandatory retirement age. Employees may request flexible working or phased retirement as they approach retirement age. The company may provide information about pension options and pre-retirement planning.",
        "source_document": "Retirement-Policy (1).docx",
    },
    {
        "question": "What conditions must be met for an employee to receive a pay increase?",
        "ground_truth": "Salary increases are not automatic. They are awarded based on performance, market rates, and company financial position. The increase amount and effective date are confirmed in a salary review letter.",
        "source_document": "Salary-Review-Letter-With-Pay-Increase.docx",
    },
    {
        "question": "What is communicated to employees when no pay increase is awarded?",
        "ground_truth": "When no pay increase is awarded, the employee receives a salary review letter confirming their current salary remains unchanged, with an explanation of the reasons (e.g., company performance, budget constraints).",
        "source_document": "Salary-Review-Letter-confirming-No-Pay-Increase.docx",
    },
    {
        "question": "How does shared parental leave work and how much can parents take?",
        "ground_truth": "Shared Parental Leave (SPL) allows eligible parents to share up to 50 weeks of leave and 37 weeks of Shared Parental Pay after the birth or adoption of a child. Leave must be taken in blocks of at least 1 week and requires advance notice.",
        "source_document": "Shared-Parental-Leave-Policy (1).docx",
    },
    {
        "question": "What is the trigger point for the sickness and absence policy's formal review process?",
        "ground_truth": "The sickness absence policy triggers a formal review when an employee reaches a set threshold, such as a certain number of days or occurrences within a rolling 12-month period. Return-to-work interviews are required after every absence.",
        "source_document": "Sickness-And-Absence-Policy.pdf",
    },
    {
        "question": "What are the minimum terms included in a statutory minimum employment offer letter?",
        "ground_truth": "A statutory minimum offer letter includes job title, start date, pay, hours of work, holiday entitlement, and notice period — the key terms required by employment law to be provided in writing within 2 months of starting.",
        "source_document": "Statutory-Minimum-Employment-Offer-Letter.docx",
    },
    # Cross-document multi-hop questions
    {
        "question": "If an employee on garden leave tests positive for drugs, what policies apply?",
        "ground_truth": "The garden leave policy governs the employee's restricted activities during the notice period, while the drugs policy applies to any positive test result regardless of employment status, potentially leading to disciplinary action.",
        "source_document": "Letter-putting-employee-on-garden-leave.docx,Drugs-Policy.pdf",
    },
    {
        "question": "How do the notice periods policy and the probationary period policy interact for employees who fail probation?",
        "ground_truth": "Employees dismissed at the end of probation are entitled to their contractual or statutory notice (whichever is greater) as defined in the notice periods policy. The probationary policy specifies the performance criteria; the dismissal letter triggers the notice entitlement.",
        "source_document": "Notice-Periods-Policy.pdf,Probationary-Periods-Policy.docx",
    },
]
