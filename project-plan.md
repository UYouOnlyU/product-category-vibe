การพัฒนา API สำหรับรายงาน CSV จาก BigQuery 

วัตถุประสงค์: พัฒนา API เพื่อสร้างไฟล์ CSV ที่มีข้อมูลจาก BigQuery โดยมีการวิเคราะห์ประเภทสินค้าด้วย AI และจัดเก็บไฟล์ที่สร้างขึ้นใน Google Cloud Storage 

รายละเอียดการทำงาน: 
Input: API รับค่าเดือนและปี (เช่น 09-2025) จากส่วนหน้า (frontend) 
Output: API จะสร้างไฟล์ CSV ที่ประกอบด้วยข้อมูลจาก BigQuery และเพิ่ม 3 คอลัมน์ใหม่ ได้แก่: ประเภทสินค้าที่ AI แนะนำ, คะแนนความเกี่ยวข้องที่ AI ให้ และเหตุผลในการแนะนำ 

ขั้นตอนการประมวลผล: 
ระบบจะดึงข้อมูลจากตาราง BigQuery โดยกรองข้อมูลจาก schema check_invoice_date ตามค่าเดือนและปีที่ได้รับ 
นำข้อมูลจาก schema item_description ไปประมวลผลด้วย Gemini 2.5 Flash Lite เพื่อแนะนำประเภทสินค้า ประเภทสินค้าที่ใช้ในการอ้างอิงทั้งหมดจะถูกเก็บไว้ในไฟล์ allowed_categories.json 
Gemini จะให้คะแนนความเกี่ยวข้องและเหตุผลประกอบการแนะนำ 
สร้างไฟล์ CSV ตามรูปแบบที่กำหนด 
จัดเก็บไฟล์ CSV ที่สร้างขึ้นไว้ใน Google Cloud Storage 

การตั้งค่าในไฟล์ .env: 
Vertex AI Settings: GCP_PROJECT_ID="loxley-orbit-dev-wan" 
GCP_LOCATION="us-central1" 
GEMINI_MODEL="gemini-2.5-flash-lite" 
Service Account: GOOGLE_APPLICATION_CREDENTIALS=service-account.json 
BigQuery Settings: TABLE_ID=loxley-orbit-dev-wan.MBTH_test.MBTH_hotel_invoice 
Path to allowed categories JSON file: CATEGORIES_PATH=allowed_categories.json 
GCS Settings: GCS_BUCKET=suwipoo-test 
GCS_OUTPUT_PREFIX=MBTH/product-category 

หมายเหตุ: 
มีการใช้ Gemini Model จาก Vertex AI 
ต้องสร้างไฟล์ Python สำหรับทดสอบ API นี้ 
API มีค่า option เป็น limit เพื่อใช้ทดสอบข้อมูลขนาดเล็ก
จะดึงข้อมูลจากตาราง BigQuery ครั้งละประมาณ 100,000 แถว