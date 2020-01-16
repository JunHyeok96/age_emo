# people_age_emo_id

demo.py

1. 웹캡 or 영상속 얼굴을 인식합니다.

<img src="https://user-images.githubusercontent.com/52908154/72505874-22dd7a00-3884-11ea-91f3-f60af55a2d5e.png" width="20%", height="20%>

2. 인식한 얼굴의 나이, 성별, 감정을 분석합니다.

![Screenshot from 2020-01-16 17-09-12](https://user-images.githubusercontent.com/52908154/72505919-34268680-3884-11ea-8e92-bc4b1036bba7.png)


3. 미리 저장되어있는 이미지와 비교해 기존의 사용자인지 비교 분석합니다.



4. 기존의 유저로 판정되면 해당 유저의 폴더에 n초 간격으로 새로운 이미지를 저장합니다.

![Screenshot from 2020-01-16 17-09-59](https://user-images.githubusercontent.com/52908154/72505988-5ae4bd00-3884-11ea-8dc4-943cbde1c665.png)

5. 새로운 유저라면 새로운 폴더를 생성하고 n초 간격으로 새로운 이미지를 저장합니다.

![Screenshot from 2020-01-16 17-11-02](https://user-images.githubusercontent.com/52908154/72506018-6932d900-3884-11ea-8d0f-a1009944882d.png)

![Screenshot from 2020-01-16 17-11-13](https://user-images.githubusercontent.com/52908154/72506149-b44cec00-3884-11ea-92c9-5123857a3a46.png)


*폴더당 이미지의 최대 갯수는 100장이며 초과시 오래된 이미지를 부터 삭제됩니다.

7. 프로그램이 종료되면 기존의 유저와 새로운 유저 폴더 내용을 비교합니다. 우발적으로 생성된 기존 유저 이미지를 기존 폴더에 병합합니다.

![Screenshot from 2020-01-16 17-12-51](https://user-images.githubusercontent.com/52908154/72506232-dba3b900-3884-11ea-9b1c-1e6a50f25d22.png)
(같은 사용자지만 신규유저로 생성된 폴더)

![Screenshot from 2020-01-16 17-12-55](https://user-images.githubusercontent.com/52908154/72506294-f413d380-3884-11ea-8761-0ccd9dd0e5b3.png)
(새로운 폴더에 이미지가 저장되는 모습)

![Screenshot from 2020-01-16 17-17-08](https://user-images.githubusercontent.com/52908154/72506336-068e0d00-3885-11ea-97de-459c0534bff4.png)
(프로그램이 종료되고 기존의 유저 폴더에 병합됨)
