# Live test results — 31 October 2026

**17/17 cases passed manual review:** 14 supported answers and 3 correct refusals.
Actual local inference used Qwen2.5-3B-Instruct Q4_K_M with neural MiniLM embeddings.
Answers below are the actual returned outputs. They are distinct from reference answers.
The sentence-boundary check retains source conditions when a model extract ends early.

| # | Question | Returned source | Actual answer | Review |
| --- | --- | --- | --- | --- |
| 1 | What are the total bootcamp fees? | Page 16 | Total Fees: R 29,950 | PASS |
| 2 | When must fees be paid in full? | Page 16 | Fees must be paid in full before the orientation day, which is scheduled for 30 March 2026. | PASS |
| 3 | What minimum internet speeds are recommended? | Page 5 | a high-speed internet connection is essential. Recommended minimum speeds are 10 Mbps for download and 3 Mbps for upload. | PASS |
| 4 | When are live classes held? | Page 11 | We will have live classes every 6 pm - 8 pm on Wednesdays and 9 am - 11 am on Mondays for the first 12 weeks of the bootcamp which will then reduce to 1 live class every Monday at 9 am - 11 am. | PASS |
| 5 | Where must assignments be submitted? | Page 13 | Google classroom | PASS |
| 6 | How much does the final project contribute to the final grade? | Page 14 | 35% | PASS |
| 7 | Who is the lead instructor? | Page 15 | Kago Kola | PASS |
| 8 | Who is my dedicated tutor? | Page 20 | Dave | PASS |
| 9 | Which channel is used for communication after orientation? | Page 18 | Discord | PASS |
| 10 | What do green calendar tiles mean? | Page 9 | Green Tiles (Same Day Completion): Demonstrates your ability to meet deadlines promptly, showcasing a strong work ethic and commitment to timely task completion. | PASS |
| 11 | What is the minimum SSD storage? | Page 5 | A solid-state drive (SSD) with at least 256 GB of storage is recommended for faster data access and improved performance. | PASS |
| 12 | When are tutor support hours? | Page 20 | Tutors will host support hours every Tuesdays 2pm to 4 pm and 6 pm to 8 pm and Thursdays at 10am to 12 pm and 6 pm to 8 pm | PASS |
| 13 | Which financing partners are available? | Page 16 | Financing with Capitec and Manati | PASS |
| 14 | What does a red calendar tile mean? | Page 9 | Red Tiles (Incomplete) | PASS |
| 15 | What is the capital of Japan? | None (refusal) | I could not find that information in the available knowledge base. | PASS |
| 16 | What is the attendance requirement? | None (refusal) | I could not find that information in the available knowledge base. | PASS |
| 17 | What is the refund policy? | None (refusal) | I could not find that information in the available knowledge base. | PASS |

## Review notes

1. Correct total fee: R 29,950.
2. Includes both the before-orientation condition and 30 March 2026 date.
3. Correct minimum speeds: 10 Mbps download, 3 Mbps upload.
4. Includes the first 12 weeks and the subsequent Monday-only schedule.
5. Correct submission platform: Google Classroom.
6. Correct final-project weighting: 35%.
7. Correct staff member and role: Kago Kola, Lead Instructor.
8. Correct dedicated tutor: Dave.
9. Correct post-orientation communication channel: Discord.
10. Correct interpretation: same day completion.
11. Correct recommended minimum SSD: 256 GB.
12. Includes both Tuesday slots and both Thursday slots.
13. Correct financing partners: Capitec and Manati.
14. Correct interpretation: incomplete.
15. Correct refusal: unrelated geography is absent.
16. Correct refusal: class times do not establish an attendance requirement.
17. Correct refusal: no refund policy is specified.

All page citations use physical PDF pages. The attendance question retrieves related
class information but correctly refuses to invent a policy. Geography and refunds
have no qualifying retrieval and are refused without invoking the LLM.

The JSON report preserves actual model outputs, chunk identifiers and validated answers.
A finite test set does not establish correctness for arbitrary future questions.
These results were recorded on 31 October 2026.
Run the commands in README.md to produce results on your own computer.
