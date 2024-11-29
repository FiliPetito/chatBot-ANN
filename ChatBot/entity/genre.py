from sqlalchemy import Column, Integer, String
from dbConnection import Base

class Genre(Base):
    __tablename__ = "genres"  # Nome della tabella nel database

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(String(20000), unique=True, nullable=False)

    def __repr__(self):
        return f"<Genre(id={self.id}, name='{self.name}', description='{self.description}')>"
