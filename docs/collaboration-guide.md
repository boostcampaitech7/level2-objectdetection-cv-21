# 협업 가이드
1. Github Flow 활용
    + **main 브랜치에는 직접 commit하지 않음.**
    + 필요한 기능이나 hotfix가 있을 경우, Github `issue` 생성함.
    + `feature-N` 브랜치(N: 생성된 이슈 숫자)를 생성하고 해당 브랜치에서 작업함.
    + 작업이 완료되면 push함.
    + `Pull Requests`를 날려 main 브랜치에 merge함.
2. 커밋 컨벤션- VSCode extension의 `Gitmoji` 이용
    + [Gitmoji 약속](https://gitmoji.dev/)
    + [Gitmoji 설치](https://inpa.tistory.com/entry/GIT-%E2%9A%A1%EF%B8%8F-Gitmoji-%EC%82%AC%EC%9A%A9%EB%B2%95-Gitmoji-cli)
    + 커밋 컨벤션 규칙을 빡빡하게 지키자는 것은 아니고 쓰는 것 자체에 의의를 두는 것으로 함.
3. 하루에 GPU 20시간 돌리기
4. 자기 전 GPU 서버가 노는지 체크하기
5. 공통 vscode extensions: `Git graph`, `pylint`, `Github Pull Requests (선택사항)`
6. 이틀에 1개씩 가설을 적극적으로 세우기
7. 말이 아프면 그 즉시 지적하기
8. 그날 각 GPU 서버 사용자는 사다리타기로 뽑음.
9. Issue와 PR을 만들 때, 수정된 [label](https://github.com/boostcampaitech7/level2-objectdetection-cv-21/labels?sort=name-asc) 이용하기
    + Gitmoji+Name : Issue의 목적에 따라서 구분된 label (ex. 🐞: Bug)
    + Issue의 중요도에 따라서 `Priority : High-Medium-Low`로 구분함.
    + `🙋‍♂️ Hypothesis` : 가설 설정 및 검증을 위한 label, "이왕이면 모든 업무를 github 내에서 진행하는 것이 좋지 않을까?" 생각해서 추가